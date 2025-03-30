GPU_ID=0
PATH_CKPT='path/to/best/fine-tuned/model'

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
--config cfgs/finetune_modelnet.yaml \
--test \
--exp_name output_file_name \
--ckpts $PATH_CKPT