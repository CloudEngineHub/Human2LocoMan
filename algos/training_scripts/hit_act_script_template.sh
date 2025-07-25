export OMP_NUM_THREADS=1

python -u algos/train_hit_or_act.py \
    --ckpt_dir your/desired/ckpt/dir \
    --dataset_dir your/dataset/dir \
    --policy_class <HIT or ACT> \
    --task_name task_name_for_saving_results \
    --train_ratio 0.95 \
    --min_val_num desired_val_file_num \
    --batch_size 24 \
    --lr 2e-5 \
    --seed 6 \
    --hidden_dim 128 \
    --num_steps 100000 \
    --validate_every 1000 \
    --save_every 5000 \
    --chunk_size 120 \
    --same_backbones \
    --feature_loss_weight 0.001 \
    --backbone resnet18 \
    --width 640 \
    --height 480 \
    --wandb \
    --wandb_name name_for_wandb_experiment
