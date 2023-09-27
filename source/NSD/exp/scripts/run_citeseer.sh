#!/bin/sh

python -m exp.run \
    --dataset=citeseer \
    --d=3 \
    --layers=4 \
    --hidden_channels=50 \
    --left_weights=True \
    --right_weights=True \
    --lr=0.05 \
    --weight_decay=5e-3 \
    --input_dropout=0.0 \
    --dropout=0.5 \
    --use_act=True \
    --model=BundleSheaf \
    --normalised=True \
    --sparse_learner=True \
    --entity="${ENTITY}"