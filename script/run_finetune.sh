GPU_ID=0
PATH_CKPT='path/to/pre-trained/model'

# ScanObjectNN
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
#--config cfgs/finetune_scan_hardest.yaml \
#--finetune_model \
#--exp_name output_file_name \
#--ckpts $PATH_CKPT

# ModelNet40 1K
 CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
 --config cfgs/finetune_modelnet.yaml \
 --finetune_model \
 --exp_name output_file_name \
 --ckpts $PATH_CKPT