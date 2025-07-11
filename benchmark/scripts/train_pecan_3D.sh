seeds=($(seq 1 16))
train_folder=/home/athenes/Paraplume/benchmark/pecan/train_set
val_folder=/home/athenes/Paraplume/benchmark/pecan/val_set
test_folder=/home/athenes/Paraplume/benchmark/pecan/test_set
train_csv=/home/athenes/Paraplume/datasets/pecan/train_set.csv
val_csv=/home/athenes/Paraplume/datasets/pecan/val_set.csv
test_csv=/home/athenes/Paraplume/datasets/pecan/test_set.csv
result_folder=/home/athenes/Paraplume/benchmark/pecan/3D/
pdb_folder_path=/home/athenes/all_structures/imgt_renumbered_pecan
for seed in "${seeds[@]}" ; do
    python graph_extension/train_graph_light.py $train_folder $val_folder $train_csv $val_csv \
    --seed $seed --patience 300 -n 300 -r $result_folder/$seed --num-workers 16\
    --gpu 0 --pdb-folder-path-train $pdb_folder_path \
    --pdb-folder-path-val $pdb_folder_path --override
    python graph_extension/evaluate_graph.py $result_folder/$seed $test_folder $test_csv --pdb-folder-path-test $pdb_folder_path
    python graph_extension/predict_graph.py $result_folder/$seed/graph_checkpoint.pt $test_folder $test_csv --pdb-folder-path $pdb_folder_path
done
