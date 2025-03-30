GPU_ID=0

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
#--config cfgs/finetune_scan_hardest.yaml \
#--scratch_model \
#--exp_name output_file_name

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
--config cfgs/finetune_modelnet.yaml \
--scratch_model \
--exp_name output_file_name