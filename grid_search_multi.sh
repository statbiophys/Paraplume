#!/bin/bash

train=/home/gathenes/paragraph_benchmark/241026_expanded/train_set
val=/home/gathenes/paragraph_benchmark/241026_expanded/val_set
test=/home/gathenes/paragraph_benchmark/241026_expanded/test_set
folder=/home/gathenes/paragraph_benchmark/241026_expanded
lrs=("0.00002" )
batch_norms=("")
dropouts=("0.2,0.2")
maskprobs=("0.4")
batchsizes=("20" "30")
dims=("2000,1000" "4000,2000")
alphas=("4.5")
seeds=("0")
l2_pens=("0")
weights=("1")
bigembeddings=("--bigembedding" )

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
                                    echo "learning rate" $lr "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "alpha" $alpha "pen" $pen "weight"  $weight $bg
                                    result_folder=$folder/lr-${lr}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_alpha-${alpha}_pen-${pen}_weight_${weight}${bg}_multi
                                    mkdir -p $result_folder
                                    python train.py $train $val -n 500 --lr $lr --dropouts $dr --mask-prob $mk -bs $bs --dims $d1 -r $result_folder --alpha $alpha --l2-pen $pen --pos-weight $weight $bg --multiobjective > $result_folder/train_logs.txt
                                    python evaluate.py $result_folder $test --multiobjective $bg > $result_folder/evaluate_logs.txt
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
