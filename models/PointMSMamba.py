from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from knn_cuda import KNN
from .block import Block
from .build import MODELS
from .serialization import Point


def serialization(pos, feat=None, x_res=None, order="z", grid_size=0.02):
    bs, n_p, _ = pos.size()
    if not isinstance(order, list):
        order = [order]

    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord

    batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)

    point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
    point_dict = Point(**point_dict)
    point_dict.serialization(order=order)

    order = point_dict.serialized_order
    inverse_order = point_dict.serialized_inverse

    pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if feat is not None:
        feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()

    return pos, order, inverse_order, feat, x_res


def serialization_func(p, x, x_res, order):
    p, order, inverse_order, x, x_res = serialization(p, x, x_res=x_res, order=order, grid_size=0.02)
    return p, order, inverse_order, x, x_res


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        if in_c == 3:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, out_c, 1)
            )

        else:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, in_c, 1),
                nn.BatchNorm1d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_c, in_c, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(in_c * 2, out_c, 1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_c, out_c, 1)
            )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.out_c)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
            idx : B G M
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # import ipdb; ipdb.set_trace()
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, idx


class EdgeGraphModule(nn.Module):
    def __init__(self, dim, k_group_size):
        super().__init__()
        self.k = k_group_size
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )

    def _knn(self, x, k):
        """
        x: Tensor of shape (B, C, N)
        返回每个点的 k 个近邻索引，输出 shape: (B, N, k)
        """
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def _get_graph_feature(self, x, k, idx=None):
        """
        x: Tensor of shape (B, C, N)
        返回拼接后的局部特征，输出 shape: (B, 2*C, N, k)
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        # 保证 x shape 为 (B, C, N)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = self._knn(x, k=k)

        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base  # (B, N, k)
        idx = idx.view(-1)

        _, num_dims, _ = x.size()  # num_dims = C

        x = x.transpose(2, 1).contiguous()  # (B, N, C)
        feature = x.view(batch_size * num_points, -1)[idx, :]  # (B*N*k, C)
        feature = feature.view(batch_size, num_points, k, num_dims)  # (B, N, k, C)

        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (B, N, k, C)
        # 拼接邻域特征差值和原始特征：差值 (feature - x) 和 x
        feature = torch.cat((feature - x, x), dim=3)  # (B, N, k, 2*C)
        feature = feature.permute(0, 3, 1, 2).contiguous()  # (B, 2*C, N, k)
        return feature

    def forward(self, x, center):
        """
        x: 输入点云特征，原始形状 (B, dim, N_points)，例如 (B, 384, 64)
        center: 中心点，形状 (B, dim_center, N_center)，例如 (B, 384, 3)
        输出: 处理后的特征，形状 (B, N_center, dim)
        """
        idx = self._knn(center.permute(0, 2, 1), k=self.k)  # (B, N_center, k)
        x_graph = self._get_graph_feature(x.permute(0, 2, 1), k=self.k, idx=idx)  # (B, 2*dim, N_center, k)

        x_conv = self.conv1(x_graph)  # (B, dim//downrate, N_center, k)
        x_pool = x_conv.max(dim=-1, keepdim=False)[0]  # (B, dim//downrate, N_center)
        x_out = self.conv2(x_pool)  # (B, dim, N_center)
        x_out = x_out.permute(0, 2, 1)  # (B, N_center, dim)
        return x_out


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        k_group_size,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    mixer_cls2 = partial(EdgeGraphModule, k_group_size=k_group_size)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        mixer_cls2,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            k_group_size: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out: int = 0.,
            drop_path: int = 0.,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    k_group_size,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, center=None, inference_params=None):
        hidden_states = input_ids + pos
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, center, inference_params=inference_params
            )
            hidden_states = self.drop_out(hidden_states)

        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))

        return hidden_states


@MODELS.register_module()
class PointMSMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.depths = config.depths
        self.cls_dim = config.cls_dim
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.trans_dims = config.trans_dims
        self.encoder_dims = config.encoder_dims
        self.k_group_size = config.k_group_size
        self.feat_dim = sum(self.trans_dims)

        self.group_divider_list = nn.ModuleList()
        self.encoder_list = nn.ModuleList()
        self.pos_embed_list = nn.ModuleList()
        self.blocks = nn.ModuleList()

        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out = 0. if not hasattr(self.config, "drop_out") else self.config.drop_out

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.alpha, std=.02)
        nn.init.normal_(self.beta, std=.02)

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, sum(self.depths))]
        for i in range(len(self.group_sizes)):
            self.group_divider_list.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

            if i == 0:
                self.encoder_list.append(Encoder(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.encoder_list.append(Encoder(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))

            self.pos_embed_list.append(nn.Sequential(
                nn.Linear(3, self.trans_dims[i]),
                nn.GELU(),
                nn.Linear(self.trans_dims[i], self.trans_dims[i]),
            ))

            self.blocks.append(MixerModel(d_model=self.trans_dims[i],
                                          n_layer=self.depths[i],
                                          k_group_size=self.k_group_size,
                                          rms_norm=self.rms_norm,
                                          drop_out=self.drop_out,
                                          drop_path=dpr[depth_count: depth_count + self.depths[i]]))
            depth_count += self.depths[i]

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('h_encoder'):
                    base_ckpt[k[len('h_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        x_list = []
        for i in range(len(self.group_divider_list)):
            if i == 0:
                neighborhood, center, idx = self.group_divider_list[i](pts)
                group_input_tokens = self.encoder_list[i](neighborhood)  # B G C
            else:
                neighborhood, center, idx = self.group_divider_list[i](center)
                b, g1, _ = x.shape  # B,512,96
                b, g2, k2, _ = neighborhood.shape  # B,256,8,3
                x_neighborhoods = x.reshape(b * g1, -1)[idx, :].reshape(b, g2, k2, -1)  # B,256,8,96
                group_input_tokens = self.encoder_list[i](x_neighborhoods)  # B,256,192

            pos = self.pos_embed_list[i](center)
            # reordering strategy
            center, _, _, group_input_tokens, pos = serialization_func(center, group_input_tokens, pos, 'z-trans')
            x = group_input_tokens
            x = self.blocks[i](x, pos, center)

            x_list.append(x.mean(1))

        global_feats1 = self.alpha * x_list[0]
        global_feats2 = self.beta * x_list[1]
        global_feats3 = x_list[2]
        x_concat = torch.cat((global_feats1, global_feats2, global_feats3), dim=-1)

        ret = self.cls_head_finetune(x_concat)
        return ret