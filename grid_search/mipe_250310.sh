#!/bin/bash
train=/home/athenes/benchmark2/mipe/train
val=/home/athenes/benchmark2/mipe/val
test=/home/athenes/benchmark2/mipe/test_set
test_csv="/home/athenes/paratope_model/datasets/mipe/test_set.csv"
pdb_folder="/home/athenes/all_structures/imgt_renumbered_mipe"

folder=/home/athenes/benchmark2/mipe/250310/
lrs=("0.00005")
dropouts=("0.4,0.4,0.4" "0.6,0.6,0.6")
batchsizes=("16")
dims=("4000,2000,1000")
seeds=($(seq 1 5))
l2_pens=("0.00001")
weights=("1")
alphas=("4,5,6")
maskprobs=("0.4" "0.6")
embeddings=("all")
cvs=("0" "1" "2" "3" "4")

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
                                        for cv in "${cvs[@]}" ; do
                                            echo "learning rate" $lr "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "alphas" $alpha "pen" $pen "weight"  $weight "seed" $seed "emb" $emb "cv" $cv
                                            result_folder=$folder/lr-${lr}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_alphas-${alpha}_pen-${pen}_weight_${weight}_emb_${emb}_seed_$seed/$cv
                                            mkdir -p $result_folder
                                            python train.py ${train}_${cv} ${val}_${cv} --gpu 0 -n 500 --lr $lr --dropouts $dr --mask-prob $mk -bs $bs --dims $d1 -r $result_folder --alphas $alpha --l2-pen $pen --pos-weight $weight --seed $seed --emb-models $emb --patience 20 > $result_folder/train_logs.txt
                                            python predict.py $result_folder/checkpoint.pt $test $test_csv --pdb-folder-path $pdb_folder --name new
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
