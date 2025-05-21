#!/bin/bash

> $1.log

for arch in GCN GAT GIN; do
    for l in 1 2 3 4 5; do
        python -u train/train_node_classification.py --device $4 --directory $3 --root $2 --dataset $1 --architecture $arch --layers $l >> $1.log
        python -u explain/explain_node_classification.py --device $4 --directory $3 --root $2 --dataset $1 --architecture $arch --layers $l >> $1.log
        python -u track_grads/tracking_gradients.py --device $4 --directory $3 --root $2 --dataset $1 --architecture $arch --layers $l >> $1.log
    done
done