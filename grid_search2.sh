#!/bin/bash

train=/home/gathenes/paragraph_benchmark/241013/train_set
val=/home/gathenes/paragraph_benchmark/241013/val_set
test=/home/gathenes/paragraph_benchmark/241013/test_set
folder=/home/gathenes/paragraph_benchmark/241013
lrs=("0.00005")
batch_norms=("--batch-norm")
dropouts=("0.2")
maskprobs=("0.4")
batchsizes=("10")
dim1s=("2000")
dim2s=("1000")
alphas=("7")

# Loop through the array
for lr in "${lrs[@]}"; do
    for bn in "${batch_norms[@]}" ; do
        for dr in "${dropouts[@]}"; do
            for mk in "${maskprobs[@]}" ; do
                for bs in "${batchsizes[@]}" ; do
                    for d1 in "${dim1s[@]}" ; do
                        for d2 in "${dim2s[@]}" ; do
                            for alpha in "${alphas[@]}" ; do
                                echo "learning rate" $lr "batchnorm" $bn "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "dim2" $d2 "alpha" $alpha
                                result_folder=$folder/lr-${lr}_bn-${bn}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_dim2-${d2}_alpha-${alpha}
                                mkdir -p $result_folder
                                python train.py $train $val -n 500 --lr $lr $bn --dropout $dr --mask-prob $mk -bs $bs --dim1 $d1 --dim2 $d2 -r $result_folder --alpha $alpha > $result_folder/train_logs.txt
                                python evaluate.py $result_folder/checkpoint.pt $test --dim1 $d1 --dim2 $d2 $bn --alpha $alpha > $result_folder/evaluate_logs.txt
                            done
                        done
                    done
                done
            done
        done
    done
done
