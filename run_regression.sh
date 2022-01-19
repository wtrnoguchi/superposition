#!/bin/bash

seed=0
test_batch_size=100

exp_config=exp1_mse
data_name=self_random_other_stay

rootdir="$(echo $PWD)"

echo "Peforming regression analysis."

for restore_epoch in {0..200}; do
    python3 test.py \
        --exp_config $exp_config \
        --seed $seed \
        --cudnn_deterministic \
        --test_epoch $restore_epoch \
        --test_data_name $data_name \
        --test_batch_size $test_batch_size \
        --save_targets self_motion self_position other_motion other_position state \
        --data_load_memory \
        --mask_off

    resultdir=$rootdir/data/result/
    saveddir=$resultdir/$exp_config/$seed/test/$data_name/save
    datadir=$rootdir/data/data/$data_name
    
    cd $saveddir
    
    ln -s $rootdir/analyze ./
    ln -s $datadir/data.h5 ./

    python3 analyze/regression.py --epoch $restore_epoch
    rm ./saved.h5

    cd $rootdir

done

cd $saveddir
python3 analyze/plot_regression.py

echo "Done."