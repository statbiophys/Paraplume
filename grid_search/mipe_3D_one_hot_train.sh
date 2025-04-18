seeds=($(seq 1 16))
train_folder=/home/athenes/benchmark2/mipe/train_val
val_folder=/home/athenes/benchmark2/mipe/test_set
test_folder=/home/athenes/benchmark2/mipe/test_set
train_csv=/home/athenes/paratope_model/datasets/mipe/train_val.csv
val_csv=/home/athenes/paratope_model/datasets/mipe/test_set.csv
test_csv=/home/athenes/paratope_model/datasets/mipe/test_set.csv
result_folder=/home/athenes/benchmark2/3D_mipe/one-hot/
pdb_folder_path=/home/athenes/all_structures/imgt_renumbered_mipe
for seed in "${seeds[@]}" ; do
    python train_graph_light.py $train_folder $val_folder $train_csv $val_csv \
    --seed $seed --patience 150 -n 150 -r $result_folder/$seed --num-workers 16\
    --gpu 1 --pdb-folder-path-train $pdb_folder_path \
    --pdb-folder-path-val $pdb_folder_path
    python evaluate_graph.py $result_folder/$seed $test_folder $test_csv --pdb-folder-path-test $pdb_folder_path
done
