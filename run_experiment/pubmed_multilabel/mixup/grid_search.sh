#!/bin/bash -i

OUTPUT="all_results"
DATASET="pubmed_multilabel"
# model state
STATE="finetune+naivemix"
# training args
LRS=(5e-6 1e-5 3e-5 5e-5)
ALPHAS=(0.1 0.5 1.0 2.0 4.0 8.0)
TRAIN_BATCH_SIZE=64
for LR in ${LRS[@]}; do
    for ALPHA in ${ALPHAS[@]}; do
        python baselines.py \
            --model_name_or_path "bert-base-uncased" \
            --model_state $STATE \
            --alpha $ALPHA \
            --output_dir "${OUTPUT}/${DATASET}/${STATE}/alpha${ALPHA}_LR${LR}" \
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