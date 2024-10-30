#!/bin/bash

train=/home/gathenes/paragraph_benchmark/241012_alphatests/train_set
val=/home/gathenes/paragraph_benchmark/241012_alphatests/val_set
test=/home/gathenes/paragraph_benchmark/241012_alphatests/test_set
folder=/home/gathenes/paragraph_benchmark/241012_alphatests
alphas=("3" "3.5" "4" "4.5" "5" "5.5" "6" "6.5" "7" "7.5")

# Loop through the array
for alpha in "${alphas[@]}"; do
    echo $alpha
    result_folder=$folder/$alpha
    mkdir -p $result_folder
    python train_new.py $train $val -r $result_folder --lr 0.0001 --alpha $alpha -n 100 --override
done
