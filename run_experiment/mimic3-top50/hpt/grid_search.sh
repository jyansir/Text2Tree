#!/bin/bash -i

LRS=(5e-6 1e-5 3e-5 5e-5)

for LR in ${LRS[@]}; do
    python htc_baselines/hpt/train.py \
        --arch "bert-base-uncased" \
        --data "mimic3-top50" \
        --seed 42 \
        --max_length 512 \
        --lr $LR \
        --name "LR${LR}"            
done
