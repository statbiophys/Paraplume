seeds=($(seq 1 16))
train_folder=/home/athenes/benchmark2/paragraph/train_set
val_folder=/home/athenes/benchmark2/paragraph/val_set
test_folder=/home/athenes/benchmark2/paragraph/test_set
train_csv=/home/athenes/paratope_model/datasets/paragraph/train_set.csv
val_csv=/home/athenes/paratope_model/datasets/paragraph/val_set.csv
test_csv=/home/athenes/paratope_model/datasets/paragraph/test_set.csv
result_folder=/home/athenes/benchmark2/3D_paragraph/one-hot/
pdb_folder_path=/home/athenes/all_structures/imgt_renumbered_expanded
for seed in "${seeds[@]}" ; do
    python train_graph_light.py $train_folder $val_folder $train_csv $val_csv \
    --seed $seed --patience 300 -n 300 -r $result_folder/$seed --num-workers 16\
    --gpu 1 --pdb-folder-path-train $pdb_folder_path \
    --pdb-folder-path-val $pdb_folder_path --override
    python evaluate_graph.py $result_folder/$seed $test_folder $test_csv --pdb-folder-path-test $pdb_folder_path

done
