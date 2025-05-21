#!/bin/bash

> Graph-SST2.log

for arch in GCN GAT GIN; do
    for l in 1 2 3 4 5; do
        python -u train/train_sst2.py --device $3 --directory $2 --root $1 --dataset Graph-SST2 --architecture $arch --layers $l >> Graph-SST2.log
        python -u explain/explain_sst2.py --device $3 --directory $2 --root $1 --dataset Graph-SST2 --architecture $arch --layers $l >> Graph-SST2.log
        python -u track_grads/tracking_gradients_sst2.py --device $3 --directory $2 --root $1 --dataset Graph-SST2 --architecture $arch --layers $l >> Graph-SST2.log
    done
done