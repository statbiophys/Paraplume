#!/bin/bash

train=/home/gathenes/paragraph_benchmark/241020_pecan/train_set
val=/home/gathenes/paragraph_benchmark/241020_pecan/val_set
test=/home/gathenes/paragraph_benchmark/241020_pecan/test_set
folder=/home/gathenes/paragraph_benchmark/241020_pecan
lrs=("0.00001" "0.000005")
batch_norms=("--batch-norm")
dropouts=("0.2" "0.15")
maskprobs=("0.3")
batchsizes=("10")
dim1s=("4000")
dim2s=("2000")
dim3s=("1")
alphas=("4.5")
seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25")

# Loop through the array
for lr in "${lrs[@]}"; do
    for bn in "${batch_norms[@]}" ; do
        for dr in "${dropouts[@]}"; do
            for mk in "${maskprobs[@]}" ; do
                for bs in "${batchsizes[@]}" ; do
                    for d1 in "${dim1s[@]}" ; do
                        for d2 in "${dim2s[@]}" ; do
                            for d3 in "${dim3s[@]}" ; do
                                for alpha in "${alphas[@]}" ; do
                                    for seed in "${seeds[@]}" ; do
                                        echo "learning rate" $lr "batchnorm" $bn "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "dim2" $d2 "dim3" $d3 "alpha" $alpha "seed" $seed
                                        result_folder=$folder/lr-${lr}_bn-${bn}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_dim2-${d2}_dim3-${d3}_alpha-${alpha}_seed-${seed}
                                        mkdir -p $result_folder
                                        python train.py $train $val -n 500 --lr $lr $bn --dropout $dr --mask-prob $mk -bs $bs --dim1 $d1 --dim2 $d2 --dim3 $d3 -r $result_folder --alpha $alpha --seed $seed > $result_folder/train_logs.txt
                                        python evaluate.py $result_folder/checkpoint.pt $test --dim1 $d1 --dim2 $d2 --dim3 $d3 $bn --alpha $alpha > $result_folder/evaluate_logs.txt
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
