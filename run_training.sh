#!/bin/bash

seed=0

for exp_config in exp1_mse exp1_l1 exp3 exp2; do
    echo "Training of "$exp_config

    python3 train.py \
        --exp_config $exp_config \
        --seed $seed \
        --cudnn_deterministic \
        --data_load_memory \

    echo "Done."
done
