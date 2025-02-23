dropouts=("0" "0.1")
depths=("6" "8" "10")
l2_pens=("0" "0.00001")
linear_dims=("10,10" "16,16")
graph_distances=("10" "12")
train_folder=/home/athenes/benchmark2/paragraph/train_set
val_folder=/home/athenes/benchmark2/paragraph/val_set
test_folder=/home/athenes/benchmark2/paragraph/test_set
train_csv=/home/athenes/paratope_model/datasets/paragraph/train_set.csv
val_csv=/home/athenes/paratope_model/datasets/paragraph/val_set.csv
test_csv=/home/athenes/paratope_model/datasets/paragraph/test_set.csv
result_folder=/home/athenes/benchmark2/3D_paragraph/one-hot-grid-search/
pdb_folder_path=/home/athenes/all_structures/imgt_renumbered_expanded
num_samples=20
sample() {
    local array=("$@")
    echo "${array[RANDOM % ${#array[@]}]}"
}
for i in $(seq 1 $num_samples); do
    ld=$(sample "${linear_dims[@]}")
    dr=$(sample "${dropouts[@]}")
    depth=$(sample "${depths[@]}")
    pen=$(sample "${l2_pens[@]}")
    gd=$(sample "${graph_distances[@]}")

    python train_graph_light.py $train_folder $val_folder $train_csv $val_csv \
    --seed 1 --patience 100 -n 400 -r $result_folder/${ld}_${dr}_${depth}_${pen}_${gd} \
    --num-workers 16 --gpu 1 --pdb-folder-path-train $pdb_folder_path \
    --num-graph-layers $depth --graph-distance $gd --linear-layers-dims $ld \
    --l2-pen $pen --pdb-folder-path-val $pdb_folder_path --override
    python evaluate_graph.py $result_folder/${ld}_${dr}_${depth}_${pen}_${gd} $test_folder $test_csv --pdb-folder-path-test $pdb_folder_path
done
