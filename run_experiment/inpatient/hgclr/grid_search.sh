#!/bin/bash -i

LRS=(5e-6 1e-5 3e-5 5e-5)
LAMDAS=(0.1 0.3 0.5 1.0)
GAMMAS=(0.005 0.01 0.02 0.05)

for LR in ${LRS[@]}; do
    for LAMDA in ${LAMDAS[@]}; do
        for GAMMA in ${GAMMAS[@]}; do
            python htc_baselines/hgclr/train_hgclr_baseline.py \
                --model "bert-base-chinese" \
                --dataset "inpatient" \
                --seed 42 \
                --max_length 128 \
                --lr $LR \
                --lamb $LAMDA \
                --thre $GAMMA \
                --per_device_train_batch_size 64 \
                --name "contrast_LAMDA${LAMDA}_GAMMA${GAMMA}_LR${LR}"            
        done
    done
done
