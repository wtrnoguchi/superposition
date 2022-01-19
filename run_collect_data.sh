#!/bin/bash

mkdir data

root_dir=$PWD

cd simulation
ln -s $root_dir/data ./data

echo "Collecting data."

for config in \
    self_random_other_stay \
    self_random_other_clockwise \
    self_random_other_counter_clockwise\
    self_stay_other_random \
    grid \
    ; do

    echo $config
    python3 collect_data.py --config $config

done


python3 combine_dataset.py \
    --save_as self_random_other_stay_periodic \
    --targets \
    self_random_other_stay \
    self_random_other_clockwise \
    self_random_other_counter_clockwise

echo "Done."