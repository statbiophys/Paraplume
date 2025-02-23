#!/bin/bash

train=/home/athenes/benchmark/paragraph/train_set
val=/home/athenes/benchmark/paragraph/val_set
test=/home/athenes/benchmark/paragraph/test_set
folder=/home/athenes/benchmark/paragraph/241210/
lrs=("0.0005")
batch_norms=("")
dropouts=("0.4,0.3,0.2")
maskprobs=("0.4")
batchsizes=("15")
dims=("4000,2000,1000")
seeds=($(seq 1 16))
l2_pens=("0")
weights=("1")
convexs=("")
distances=("")
alphas=("-" "5" "4" "4,5,6" "4,5,5.5,6,6.5" "3,3.5,4,5,5.5,6,6.5,7,7.5")
embeddings=("igT5" "antiberty" "ablang2" "igbert" "esm" "prot-t5" "all")

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
                                                result_folder=$folder/lr-${lr}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_alphas-${alpha}_pen-${pen}_weight_${weight}_multi_${cv}_${ds}_emb_${emb}_seed_$seed
                                                mkdir -p $result_folder
                                                python train.py $train $val -n 500 --lr $lr --dropouts $dr --mask-prob $mk -bs $bs --dims $d1 -r $result_folder --alphas $alpha --l2-pen $pen --pos-weight $weight $cv $ds --seed $seed --emb-models $emb --patience 6 > $result_folder/train_logs.txt
                                                python evaluate.py $result_folder $test > $result_folder/evaluate_logs.txt
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
