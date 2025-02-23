#!/bin/bash

train=/home/athenes/benchmark/paragraph/train_set
val=/home/athenes/benchmark/paragraph/val_set
test=/home/athenes/benchmark/paragraph/test_set
train_csv=/home/athenes/paratope_model/datasets/paragraph/train_set.csv
val_csv=/home/athenes/paratope_model/datasets/paragraph/val_set.csv
test_csv=/home/athenes/paratope_model/datasets/paragraph/test_set.csv
folder=/home/athenes/benchmark/graph/paragraph_one_hot_241226/
pdb_folder_path=/home/athenes/all_structures/imgt_renumbered_expanded
lrs=("0.001")
dropouts=("0")
maskprobs=("0")
depths=("6")
l2_pens=("0")
linear_dims=("10,10")
graph_distances=("10")
embeddings=("one-hot")
weights=("3.0")

# Number of random configurations to sample
num_samples=20

# Function to sample a random element from an array
sample() {
    local array=("$@")
    echo "${array[RANDOM % ${#array[@]}]}"
}

# Loop through random samples
for seed in $(seq 1 16); do
    lr=$(sample "${lrs[@]}")
    ld=$(sample "${linear_dims[@]}")
    dr=$(sample "${dropouts[@]}")
    mk=$(sample "${maskprobs[@]}")
    depth=$(sample "${depths[@]}")
    pen=$(sample "${l2_pens[@]}")
    emb=$(sample "${embeddings[@]}")
    gd=$(sample "${graph_distances[@]}")
    w=$(sample "${weights[@]}")

    echo "Sample $seed: learning rate $lr, dropout $dr, maskprob $mk, depth $depth, l2_pen $pen, graph distance $gd, embedding $emb weight $w linear dims $ld"

    result_folder=$folder/lr-${lr}_dr-${dr}_mk-${mk}_depth-${depth}_pen-${pen}_emb_${emb}_lin_dim-${ld}_graph_distance-${gd}_weight-${w}_seed-${seed}
    mkdir -p $result_folder

    python train_graph.py $train $val $train_csv $val_csv -n 300 \
        --graph-distance $gd --lr $lr --dropout $dr \
        --mask-prob $mk --num-graph-layers $depth \
        --linear-layers-dims $ld -r $result_folder \
        --l2-pen $pen --pos-weight $w --emb-models $emb \
        --patience 300 --pdb-folder-path-train $pdb_folder_path --pdb-folder-path-val $pdb_folder_path \
        -bs 16 --seed $seed \

    python evaluate_graph.py $result_folder $test $test_csv --pdb-folder-path-test $pdb_folder_path
done
