#!/bin/bash

train=/home/athenes/benchmark/mipe/train_val
test=/home/athenes/benchmark/mipe/test_set
folder=/home/athenes/benchmark/mipe/241212/
lrs=("0.0005")
batch_norms=("")
dropouts=("0.4,0.3,0.2")
maskprobs=("0" "0.4")
batchsizes=("15")
dims=("4000,2000,1000")
seeds=($(seq 1 2))
l2_pens=("0")
weights=("1")
convexs=("")
distances=("")
alphas=("-" "5" "4" "4,5,6" "4,5,5.5,6,6.5")
embeddings=("esm" "prot-t5" "all")

# Loop through the array
for lr in "${lrs[@]}"; do
    for dr in "${dropouts[@]}"; do
        for mk in "${maskprobs[@]}" ; do
            for bs in "${batchsizes[@]}" ; do
                for d1 in "${dims[@]}" ; do
                    for alpha in "${alphas[@]}" ; do
                        for pen in "${l2_pens[@]}" ; do
                            for weight in "${weights[@]}" ; do
                                for cv in "${convexs[@]}" ; do
                                    for ds in "${distances[@]}" ; do
                                        for seed in "${seeds[@]}" ; do
                                            for emb in "${embeddings[@]}" ; do
                                                echo "learning rate" $lr "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "alphas" $alpha "pen" $pen "weight"  $weight "ds" $ds "cv" $cv "seed" $seed "emb" $emb
                                                result_folder=$folder/mk-${mk}_alphas-${alpha}_emb_${emb}_seed_$seed
                                                mkdir -p $result_folder
                                                python train.py $train $test -n 500 --lr $lr --dropouts $dr --mask-prob $mk -bs $bs --dims $d1 -r $result_folder --alphas $alpha --l2-pen $pen --pos-weight $weight $cv $ds --seed $seed --emb-models $emb --patience 6 > $result_folder/train_logs.txt
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
    done
done
