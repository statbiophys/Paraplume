seeds=($(seq 1 2))
lrs=("0.0001" "0.00001")

train_folder=/home/athenes/benchmark2/paragraph/train_set
val_folder=/home/athenes/benchmark2/paragraph/val_set
test_folder=/home/athenes/benchmark2/paragraph/test_set
train_csv=/home/athenes/paratope_model/datasets/paragraph/train_set.csv
val_csv=/home/athenes/paratope_model/datasets/paragraph/val_set.csv
test_csv=/home/athenes/paratope_model/datasets/paragraph/test_set.csv
result_folder=/home/athenes/benchmark2/3D_paragraph/esm/250308/
pdb_folder_path=/home/athenes/all_structures/imgt_renumbered_expanded
for lr in "${lrs[@]}" ; do
    python train_graph_light.py $train_folder $val_folder $train_csv $val_csv \
    --seed 1 --patience 50 -n 500 -r $result_folder/$lr --num-workers 16\
    --gpu 1 --pdb-folder-path-train $pdb_folder_path \
    --pdb-folder-path-val $pdb_folder_path --override --emb-models esm --lr $lr --pos-weight 3
    python evaluate_graph.py $result_folder/$lr $test_folder $test_csv --pdb-folder-path-test $pdb_folder_path
done
