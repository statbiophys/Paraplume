#!/bin/bash

train=/home/athenes/benchmark/mipe/train_val
val=/home/athenes/benchmark/mipe/test_set
train_csv=/home/athenes/paratope_model/datasets/mipe/train_val.csv
val_csv=/home/athenes/paratope_model/datasets/mipe/test_set.csv
folder=/home/athenes/benchmark/graph/mipe_241226/
pdb_folder_path=/home/athenes/all_structures/imgt_renumbered_mipe
lrs=("0.00001")
dropouts=("0" "0.3")
maskprobs=("0" "0.3")
depths=("6" "8" "10")
l2_pens=("0" "0.00001")
linear_dims=("10,10" "32,16")
graph_distances=("10" "12")
embeddings=("esm" "esm,prot-t5")
weights=("1.0")

# Number of random configurations to sample
num_samples=20

# Function to sample a random element from an array
sample() {
    local array=("$@")
    echo "${array[RANDOM % ${#array[@]}]}"
}

# Loop through random samples
for i in $(seq 1 $num_samples); do
    lr=$(sample "${lrs[@]}")
    ld=$(sample "${linear_dims[@]}")
    dr=$(sample "${dropouts[@]}")
    mk=$(sample "${maskprobs[@]}")
    depth=$(sample "${depths[@]}")
    pen=$(sample "${l2_pens[@]}")
    emb=$(sample "${embeddings[@]}")
    gd=$(sample "${graph_distances[@]}")
    w=$(sample "${weights[@]}")

    echo "Sample $i: learning rate $lr, dropout $dr, maskprob $mk, depth $depth, l2_pen $pen, graph distance $gd, embedding $emb weight $w"

    result_folder=$folder/lr-${lr}_dr-${dr}_mk-${mk}_depth-${depth}_pen-${pen}_emb_${emb}_lin_dim-${ld}_graph_distance-${gd}_weight-${w}
    mkdir -p $result_folder

    python train_graph_light.py $train $val $train_csv $val_csv -n 500 \
        --graph-distance $gd --lr $lr --dropout $dr \
        --mask-prob $mk --num-graph-layers $depth \
        --linear-layers-dims $ld -r $result_folder \
        --l2-pen $pen --pos-weight $w --emb-models $emb \
        --patience 75 --pdb-folder-path-train $pdb_folder_path \
        --num-workers 16 -bs 16 \
        --pdb-folder-path-val $pdb_folder_path \
        --gpu 0
done
