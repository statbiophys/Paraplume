#!/bin/bash

train=/home/gathenes/paragraph_benchmark/241012_gs/train_set
val=/home/gathenes/paragraph_benchmark/241012_gs/val_set
test=/home/gathenes/paragraph_benchmark/241012_gs/test_set
folder=/home/gathenes/paragraph_benchmark/241012_gs
lrs=("0.00005" "0.00001" "0.000005" "0.000001")
batch_norms=("" "--batch-norm")
dropouts=("0" "0.2" "0.4")
maskprobs=("0" "0.2" "0.4")
batchsizes=("5" "10" "20")
dim1s=("1000" "2000" "4000")
dim2s=("1" "1000" "2000" "4000")

# Loop through the array
for lr in "${lrs[@]}"; do
    for bn in "${batch_norms[@]}" ; do
        for dr in "${dropouts[@]}"; do
            for mk in "${maskprobs[@]}" ; do
                for bs in "${batchsizes[@]}" ; do
                    for d1 in "${dim1s[@]}" ; do
                        for d2 in "${dim2s[@]}" ; do
                            echo "learning rate" $lr "batchnorm" $bn "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "dim2" $d2
                            result_folder=$folder/lr-${lr}_bn-${bn}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_dim2-${d2}
                            mkdir $result_folder
                            python train.py $train $val -n 500 --lr $lr $bn --dropout $dr --mask-prob $mk -bs $bs --dim1 $d1 --dim2 $d2 -r $result_folder > $result_folder/train_logs.txt
                            python evaluate.py $result_folder/checkpoint.pt $test --dim1 $d1 --dim2 $d2 $bn > $result_folder/evaluate_logs.txt
                        done
                    done
                done
            done
        done
    done
done
