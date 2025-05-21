> $1\_SGC.log

for l in 1 2 3 4 5; do
    python -u train/train_sgc.py --device $4 --directory $3 --root $2 --dataset $1 --architecture SGC --layers $l >> $1\_SGC.log
    python -u explain/explain_sgc.py --device $4 --directory $3 --root $2 --dataset $1 --architecture SGC --layers $l >> $1\_SGC.log
    python -u explain/sgc_occlusion.py --device $4 --directory $3 --root $2 --dataset $1 --architecture SGC --layers $l >> $1\_SGC.log
    python -u track_grads/tracking_gradients_sgc.py --device $4 --directory $3 --root $2 --dataset $1 --architecture SGC --layers $l >> $1\_SGC.log
done