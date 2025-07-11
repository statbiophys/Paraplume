#!/bin/bash

train=/home/athenes/Paraplume/benchmark/mipe/train
val=/home/athenes/Paraplume/benchmark/mipe/val
test=/home/athenes/Paraplume/benchmark/mipe/test_set
test_csv="/home/athenes/Paraplume/datasets/mipe/test_set.csv"
pdb_folder="/home/athenes/all_structures/imgt_renumbered_mipe"

folder=/home/athenes/Paraplume/benchmark/mipe/250526

# Single value parameters assigned directly
lr="0.00005"
dr="0.4,0.4,0.4"
bs="16"
d1="2000,1000,500"
pen="0.00001"
weight="1"
alpha="4,5,6"
mk="0.4"
cvs=("0" "1" "2" "3" "4")
emb="all"
seeds=($(seq 1 5))

# Loop through seeds only
for seed in "${seeds[@]}"; do
    for cv in "${cvs[@]}"; do
        echo "learning rate" $lr "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "alphas" $alpha "pen" $pen "weight" $weight "seed" $seed "emb" $emb "cv" $cv
        result_folder=$folder/lr-${lr}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_alphas-${alpha}_pen-${pen}_weight_${weight}_emb_${emb}_seed_$seed/$cv
        mkdir -p "$result_folder"
        python paraplume/train.py ${train}_${cv} ${val}_${cv} --gpu 0 -n 500 --lr "$lr" --dropouts "$dr" --mask-prob "$mk" -bs "$bs" --dims "$d1" -r "$result_folder" --alphas "$alpha" --l2-pen "$pen" --pos-weight "$weight" --seed "$seed" --emb-models "$emb" --patience 20 > "$result_folder/train_logs.txt"
        python paraplume/predict.py "$result_folder/checkpoint.pt" "$test" "$test_csv" --pdb-folder-path "$pdb_folder"
    done
done
seeds=($(seq 1 5))
llms=("igT5" "antiberty" "ablang2" "igbert" "esm" "prot-t5")
for seed in "${seeds[@]}"; do
    for cv in "${cvs[@]}"; do
        for emb in "${llms[@]}"; do
            echo "learning rate" $lr "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "alphas" $alpha "pen" $pen "weight" $weight "seed" $seed "emb" $emb
            result_folder=$folder/lr-${lr}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_alphas-${alpha}_pen-${pen}_weight_${weight}_emb_${emb}_seed_$seed/$cv
            mkdir -p "$result_folder"
            python paraplume/train.py ${train}_${cv} ${val}_${cv} --gpu 0 -n 500 --lr "$lr" --dropouts "$dr" --mask-prob "$mk" -bs "$bs" --dims "$d1" -r "$result_folder" --alphas "$alpha" --l2-pen "$pen" --pos-weight "$weight" --seed "$seed" --emb-models "$emb" --patience 20 > "$result_folder/train_logs.txt"
            python paraplume/predict.py "$result_folder/checkpoint.pt" "$test" "$test_csv" --pdb-folder-path "$pdb_folder"
        done
    done
done
llms=(
    "igT5,antiberty,ablang2,igbert,esm"
    "igT5,antiberty,ablang2,igbert,prot-t5"
    "igT5,antiberty,ablang2,esm,prot-t5"
    "igT5,antiberty,igbert,esm,prot-t5"
    "igT5,ablang2,igbert,esm,prot-t5"
    "antiberty,ablang2,igbert,esm,prot-t5"
)
for seed in "${seeds[@]}"; do
    for cv in "${cvs[@]}"; do
        for emb in "${llms[@]}"; do
            echo "learning rate" $lr "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "alphas" $alpha "pen" $pen "weight" $weight "seed" $seed "emb" $emb
            result_folder=$folder/lr-${lr}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_alphas-${alpha}_pen-${pen}_weight_${weight}_emb_${emb}_seed_$seed/$cv
            mkdir -p "$result_folder"
            python paraplume/train.py ${train}_${cv} ${val}_${cv} --gpu 0 -n 500 --lr "$lr" --dropouts "$dr" --mask-prob "$mk" -bs "$bs" --dims "$d1" -r "$result_folder" --alphas "$alpha" --l2-pen "$pen" --pos-weight "$weight" --seed "$seed" --emb-models "$emb" --patience 20 > "$result_folder/train_logs.txt"
            python paraplume/predict.py "$result_folder/checkpoint.pt" "$test" "$test_csv" --pdb-folder-path "$pdb_folder"
        done
    done
done
