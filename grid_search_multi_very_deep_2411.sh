#!/bin/bash

train=/home/gathenes/paragraph_benchmark/241107/train_set
val=/home/gathenes/paragraph_benchmark/241107/val_set
test=/home/gathenes/paragraph_benchmark/241107/test_set
folder=/home/gathenes/paragraph_benchmark/241107/
lrs=("0.00001" "0.00005" "0.0001")
batch_norms=("")
dropouts=("0.4,0.3,0.2,0.2")
maskprobs=("0.4")
batchsizes=("20" "15" "25")
dims=("4000,2000,1000,500")
seeds=("4" "5" "6" "7" "8")
l2_pens=("0" "0.00001")
weights=("1" "1.2" "1.4")
bigembeddings=("--bigembedding")
convexs=("")
distances=("")
alphas=("3,3.5,4,5,5.5,6,6.5,7,7.5")

# Loop through the array
for lr in "${lrs[@]}"; do
    for dr in "${dropouts[@]}"; do
        for mk in "${maskprobs[@]}" ; do
            for bs in "${batchsizes[@]}" ; do
                for d1 in "${dims[@]}" ; do
                    for alpha in "${alphas[@]}" ; do
                        for pen in "${l2_pens[@]}" ; do
                            for weight in "${weights[@]}" ; do
                                for bg in "${bigembeddings[@]}" ; do
                                    for cv in "${convexs[@]}" ; do
                                        for ds in "${distances[@]}" ; do
                                            for seed in "${seeds[@]}" ; do
                                                echo "learning rate" $lr "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "alphas" $alpha "pen" $pen "weight"  $weight $bg "ds" $ds "cv" $cv "seed" $seed
                                                result_folder=$folder/zz_lr-${lr}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_alphas-${alpha}_pen-${pen}_weight_${weight}_${bg}_multi_${cv}_${ds}_seed_$seed
                                                mkdir -p $result_folder
                                                python train.py $train $val -n 500 --lr $lr --dropouts $dr --mask-prob $mk -bs $bs --dims $d1 -r $result_folder --alphas $alpha --l2-pen $pen --pos-weight $weight $bg $cv $ds --seed $seed --patience 20 > $result_folder/train_logs.txt
                                                python evaluate.py $result_folder $test $bg > $result_folder/evaluate_logs.txt
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
