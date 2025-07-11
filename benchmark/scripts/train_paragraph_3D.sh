seeds=($(seq 1 16))
train_folder=/home/athenes/Paraplume/benchmark/paragraph/train_set
val_folder=/home/athenes/Paraplume/benchmark/paragraph/val_set
test_folder=/home/athenes/Paraplume/benchmark/paragraph/test_set
train_csv=/home/athenes/Paraplume/datasets/paragraph/train_set.csv
val_csv=/home/athenes/Paraplume/datasets/paragraph/val_set.csv
test_csv=/home/athenes/Paraplume/datasets/paragraph/test_set.csv
result_folder=/home/athenes/Paraplume/benchmark/paragraph/3D/
pdb_folder_path=/home/athenes/all_structures/imgt_renumbered_expanded
for seed in "${seeds[@]}" ; do
    python graph_extension/train_graph_light.py $train_folder $val_folder $train_csv $val_csv \
    --seed $seed --patience 300 -n 300 -r $result_folder/$seed --num-workers 16\
    --gpu 1 --pdb-folder-path-train $pdb_folder_path \
    --pdb-folder-path-val $pdb_folder_path
    python graph_extension/evaluate_graph.py $result_folder/$seed $test_folder $test_csv --pdb-folder-path-test $pdb_folder_path
    python graph_extension/predict_graph.py $result_folder/$seed/graph_checkpoint.pt $test_folder $test_csv --pdb-folder-path /home/athenes/all_structures/imgt_renumbered_expanded
    python graph_extension/predict_graph.py $result_folder/$seed/graph_checkpoint.pt $test_folder $test_csv --pdb-folder-path /home/athenes/all_structures/abb3_paragraph_renumbered --name abb3
done
