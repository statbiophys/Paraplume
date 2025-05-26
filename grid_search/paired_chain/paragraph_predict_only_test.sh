#!/bin/bash

test=/home/athenes/benchmark3/paragraph/paired_chain/test_set_light
test_csv=/home/athenes/Paraplume/datasets/paragraph/test_set_light.csv
pdb_folder=/home/athenes/all_structures/imgt_renumbered_expanded

folder=/home/athenes/benchmark3/paragraph/paired_chain/250422/
lrs=("0.00005")
dropouts=("0.4,0.4,0.4")
batchsizes=("16")
dims=("2000,1000,500")
seeds=($(seq 1 16))
l2_pens=("0.00001")
weights=("1")
alphas=("4,5,6")
maskprobs=("0.4")
embeddings=("all")

# Loop through the array
for lr in "${lrs[@]}"; do
    for dr in "${dropouts[@]}"; do
        for mk in "${maskprobs[@]}" ; do
            for bs in "${batchsizes[@]}" ; do
                for d1 in "${dims[@]}" ; do
                    for alpha in "${alphas[@]}" ; do
                        for pen in "${l2_pens[@]}" ; do
                            for weight in "${weights[@]}" ; do
                                for seed in "${seeds[@]}" ; do
                                    for emb in "${embeddings[@]}" ; do
                                        echo "learning rate" $lr "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "alphas" $alpha "pen" $pen "weight"  $weight "seed" $seed "emb" $emb
                                        result_folder=$folder/lr-${lr}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_alphas-${alpha}_pen-${pen}_weight_${weight}_emb_${emb}_seed_$seed
                                        python /home/athenes/Paraplume/paraplume/paired_chain/predict.py $result_folder/checkpoint.pt $test $test_csv --pdb-folder-path $pdb_folder
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
