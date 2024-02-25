#!/bin/bash -i

OUTPUT="all_results"
DATASET="gastroenterology"
# model state
STATE="finetune+selfcon"
# training args
LRS=(5e-6 1e-5 3e-5 5e-5)
LAMDAS=(0.001 0.005 0.01 0.05)
TEMPS=(0.5 1.0 2.0 3.0)
TRAIN_BATCH_SIZE=64
for LR in ${LRS[@]}; do
    for LAMDA in ${LAMDAS[@]}; do
        for TEMP in ${TEMPS[@]}; do
            python baselines.py \
                --model_name_or_path "bert-base-chinese" \
                --model_state $STATE \
                --lamda $LAMDA \
                --temperature $TEMP \
                --output_dir "${OUTPUT}/${DATASET}/${STATE}/LAMDA${LAMDA}_TEMP${TEMP}_LR${LR}" \
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