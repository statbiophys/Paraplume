#!/bin/bash

test_heavy=/home/athenes/Paraplume/benchmark/paragraph/test_set_heavy
test_heavy_csv="/home/athenes/Paraplume/datasets/paragraph/test_set_heavy.csv"
test_light=/home/athenes/Paraplume/benchmark/paragraph/test_set_light
test_light_csv="/home/athenes/Paraplume/datasets/paragraph/test_set_light.csv"

pdb_folder="/home/athenes/all_structures/imgt_renumbered_expanded"

folder=/home/athenes/Paraplume/benchmark/paragraph/250526

# Single value parameters assigned directly
lr="0.00005"
dr="0.4,0.4,0.4"
bs="16"
d1="2000,1000,500"
pen="0.00001"
weight="1"
alpha="4,5,6"
mk="0.4"

emb="all"
seeds=($(seq 1 16))

# Loop through seeds only
for seed in "${seeds[@]}"; do
    echo "learning rate" $lr "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "alphas" $alpha "pen" $pen "weight" $weight "seed" $seed "emb" $emb
    result_folder=$folder/lr-${lr}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_alphas-${alpha}_pen-${pen}_weight_${weight}_emb_${emb}_seed_$seed
    mkdir -p "$result_folder"
    python paraplume/predict.py "$result_folder/checkpoint.pt" "$test_heavy" "$test_heavy_csv" --pdb-folder-path "$pdb_folder"
    python paraplume/predict.py "$result_folder/checkpoint.pt" "$test_light" "$test_light_csv" --pdb-folder-path "$pdb_folder"
done
