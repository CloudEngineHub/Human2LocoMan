# finetuning on a trained MXT model

export OMP_NUM_THREADS=1

python -u algos/train_mxt.py \
    --ckpt_dir your/desired/ckpt/dir \
    --dataset_dir your/dataset/dir \
    --embodiment_config_path algos/detr/models/mxt_definitions/configs/embodiments.yaml \
    --trunk_config_path algos/detr/models/mxt_definitions/configs/transformer_trunk.yaml \
    --policy_class MXT \
    --task_name task_name_for_saving_results \
    --train_ratio 0.95 \
    --min_val_num desired_val_file_num \
    --batch_size 16 \
    --lr 5e-5 \
    --lr_tokenizer 5e-5 \
    --lr_action_head 5e-5 \
    --lr_trunk 5e-5 \
    --seed 6 \
    --num_steps 60000 \
    --validate_every 1000 \
    --save_every 5000 \
    --chunk_size 60 \
    --no_encoder \
    --same_backbones \
    --feature_loss_weight 0.0 \
    --width 1280 \
    --height 480 \
    --use_wrist \
    --load_pretrain \
    --pretrained_path path/to/pretrained/mxt/policy_last.ckpt \
    --wandb \
    --wandb_name name_for_wandb_experiment