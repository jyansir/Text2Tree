#!/bin/bash -i

OUTPUT="all_results"
DATASET="inpatient"
# model state
STATE="finetune+focal"
# training args
TRAIN_BATCH_SIZE=64
LRS=(5e-6 1e-5 3e-5 5e-5)
FOCAL_ALPHAS=(0.5 0.75 0.25)
FOCAL_GAMMAS=(0.5 1.0 2.0 3.0)
for LR in ${LRS[@]}; do
    for FOCAL_ALPHA in ${FOCAL_ALPHAS[@]}; do
        for FOCAL_GAMMA in ${FOCAL_GAMMAS[@]}; do
            python baselines.py \
                --model_name_or_path "bert-base-chinese" \
                --focal_alpha $FOCAL_ALPHA \
                --focal_gamma $FOCAL_GAMMA \
                --model_state $STATE \
                --output_dir "${OUTPUT}/${DATASET}/${STATE}/alpha${FOCAL_ALPHA}_beta${FOCAL_GAMMA}_LR${LR}" \
                --dataset $DATASET \
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
    done
done
