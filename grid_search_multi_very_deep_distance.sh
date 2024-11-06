#!/bin/bash

train=/home/gathenes/paragraph_benchmark/convex_hull/train_set
val=/home/gathenes/paragraph_benchmark/convex_hull/val_set
test=/home/gathenes/paragraph_benchmark/convex_hull/test_set
folder=/home/gathenes/paragraph_benchmark/convex_hull
lrs=("0.001" "0.0001" "0.00001")
batch_norms=("")
dropouts=("0.4,0.3,0.2")
maskprobs=("0.4")
batchsizes=("15")
dims=("4000,2000,500")
seeds=("0")
l2_pens=("0")
weights=("1" "3")
bigembeddings=("--bigembedding")
convexs=("--distance")
distances=("")
alphas=("4,5,6" "4,5,5.5,6,6.5")

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
                                            echo "learning rate" $lr "dropout" $dr "maskprob" $mk "batchsize" $bs "dim1" $d1 "alphas" $alpha "pen" $pen "weight"  $weight $bg "ds" $ds "cv" $cv
                                            result_folder=$folder/z_lr-${lr}_dr-${dr}_mk-${mk}_bs-${bs}_dim1-${d1}_alphas-${alpha}_pen-${pen}_weight_${weight}_${bg}_multi_${cv}_${ds}
                                            mkdir -p $result_folder
                                            python train.py $train $val -n 500 --lr $lr --dropouts $dr --mask-prob $mk -bs $bs --dims $d1 -r $result_folder --alphas $alpha --l2-pen $pen --pos-weight $weight $bg --multiobjective $cv $ds > $result_folder/train_logs.txt
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
    done
done
