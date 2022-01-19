
seed=0
restore_epoch=200
zeropadded_epoch=`printf %05d $restore_epoch`
test_batch_size=100

echo "Performing analysis on "$1

if [ $1 = "exp1" ]; then
    exp_config=exp1_l1
    data_name=self_random_other_stay
    visualize_data_indexes=(0 1 2 3 4)
    save_target_indexes="0 1 2 3 4"
elif [ $1 = "exp3" ]; then
    exp_config=exp3
    data_name=self_random_other_stay_periodic
    visualize_data_indexes=(0 1 2 3 4  100 101 102 103 104 200 201 202 203 204)
    save_target_indexes="0 1 2 3 4 100 101 102 103 104 200 201 202 203 204"
fi

python3 test.py \
    --exp_config $exp_config \
    --seed $seed \
    --cudnn_deterministic \
    --test_epoch $restore_epoch \
    --test_data_name $data_name \
    --test_batch_size $test_batch_size \
    --save_targets self_motion self_position other_motion other_position state \
    --data_load_memory

python3 test.py \
    --exp_config $exp_config \
    --seed $seed \
    --cudnn_deterministic \
    --test_epoch $restore_epoch \
    --test_data_name $data_name \
    --test_batch_size $test_batch_size \
    --save_targets vision \
    --data_load_memory \
    --test_modes test \
    --save_vision_img \
    --img_ext png \
    --save_targets_index $save_target_indexes

rootdir="$(echo $PWD)"

resultdir=$rootdir/data/result/
saveddir=$resultdir/$exp_config/$seed/test/$data_name/save
datadir=$rootdir/data/data/$data_name

cd $saveddir

ln -s $rootdir/analyze ./
ln -s $datadir/data.h5 ./

if [ $exp_config = "exp3" ]; then

    pca_seed=0
    pca_restore_epoch=200
    pca_zeropadded_epoch=`printf %05d $restore_epoch`
    pca_exp_config=exp1_l1
    pca_data_name=self_random_other_stay

    pcadir=$resultdir/$pca_exp_config/$pca_seed/test/$pca_data_name/save/$pca_zeropadded_epoch/pca

    mkdir ./$zeropadded_epoch
    cp -r $pcadir ./$zeropadded_epoch/
fi

for mode in eval test; do
    python3 analyze/plot_state.py \
        --epoch $restore_epoch \
        --mode $mode \
        --plot self_position \
        --plot_layer self
    
    python3 analyze/plot_state.py \
        --epoch $restore_epoch \
        --mode $mode \
        --plot other_position \
        --plot_layer other
done


mode=test

if [ $exp_config = "exp3" ]; then

    python3 analyze/plot_motion.py \
        --epoch $restore_epoch \
        --mode $mode
fi


for i in "${visualize_data_indexes[@]}"; do

    python3 analyze/plot_state.py \
        --epoch $restore_epoch \
        --mode $mode \
        --plot self_position \
        --plot_layer self \
        --animation \
        --idx $i

    python3 analyze/plot_state.py \
        --epoch $restore_epoch \
        --mode $mode \
        --plot other_position \
        --plot_layer other \
        --animation \
        --idx $i

    python3 analyze/record.py \
        --epoch $restore_epoch \
        --mode $mode \
        --idx $i
done



for dir in record/overview self_vision; do

    cd $saveddir/$zeropadded_epoch/$mode
    cd $dir

    ln -s $rootdir/analyze/create_gif.py ./

    if [ $dir = "record/overview" ]; then
        gif_targets="input truth"
    elif [ $dir = "self_vision" ]; then
        gif_targets="input prediction truth"
    fi

    for i in "${visualize_data_indexes[@]}"; do

        python3 create_gif.py --idx $i --targets $gif_targets

    done
done

echo "Done."