
seed=0
restore_epoch=10
exp_config=exp2
test_batch_size=100
mode=eval
zeropadded_epoch=`printf %05d $restore_epoch`

visualize_data_indexes=(0 1 2 3 4 5 6 7 8 9)

rootdir="$(echo $PWD)"
resultdir=$rootdir/data/result/

echo "Performing analysis on exp2"

for data_name in self_stay_other_random grid; do

    cd $rootdir

    if [ $data_name = "self_stay_other_random" ]; then
        save_img='--save_vision_img --img_ext png'
    else
        save_img=''
    fi

    python3 test.py \
        --exp_config $exp_config \
        --seed $seed \
        --cudnn_deterministic \
        --test_epoch $restore_epoch \
        --test_data_name $data_name \
        --test_batch_size $test_batch_size \
        --save_targets position vision \
        --data_load_memory \
        --test_modes $mode \
        $save_img

    saveddir=$resultdir/$exp_config/$seed/test/$data_name/save
    datadir=$rootdir/data/data/$data_name

    cd $saveddir

    ln -s $rootdir/analyze ./
    ln -s $datadir/data.h5 ./


    if [ $data_name = "grid" ]; then

        python3 analyze/analyze_vpt.py --epoch $restore_epoch --mode $mode --margin 1 --self_str "a1" --other_str "a2" --save_vision

    else

        for i in "${visualize_data_indexes[@]}"; do
        
            python3 analyze/record.py \
                --epoch $restore_epoch \
                --mode $mode \
                --idx $i
        done

        for dir in record/overview self_vision other_vision; do
        
            cd $saveddir/$zeropadded_epoch/$mode
            cd $dir
        
            ln -s $rootdir/analyze/create_gif.py ./

            if [ $dir = "record/overview" ]; then
                gif_targets="input"
            elif [ $dir = "self_vision" ]; then
                gif_targets="input reconstruction"
            fi
        
            for i in "${visualize_data_indexes[@]}"; do
        
                python3 create_gif.py --idx $i --targets $gif_targets
        
            done

        done

    fi

done

echo "Done."