#!/bin/bash -i

OUTPUT="all_results"
DATASET="mimic3-top50"
# model state
STATE="finetune+dbloss"
# tuned training args
TRAIN_BATCH_SIZE=64
LRS=(5e-6 1e-5 3e-5 5e-5)
for LR in ${LRS[@]}; do
    python baselines.py \
        --model_name_or_path "bert-base-uncased" \
        --model_state $STATE \
        --output_dir "${OUTPUT}/${DATASET}/${STATE}/lr${LR}" \
        --dataset $DATASET \
        --max_length 512 \
        --learning_rate $LR \
        --lr_scheduler_type "constant" \
        --per_device_train_batch_size $TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size 128 \
        --do_train \
        --do_eval \
        --overwrite_output_dir \
        --num_train_epochs 100.0 \
        --patience 10 \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --metric_for_best_model "macro-f1"
done