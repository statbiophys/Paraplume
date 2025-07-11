seeds=($(seq 1 5))
train_folder=/home/athenes/Paraplume/benchmark/mipe/train
val_folder=/home/athenes/Paraplume/benchmark/mipe/val
test_folder=/home/athenes/Paraplume/benchmark/mipe/test_set

train=/home/athenes/Paraplume/datasets/mipe/train
val=/home/athenes/Paraplume/datasets/mipe/val

test_csv=/home/athenes/Paraplume/datasets/mipe/test_set.csv
result_folder=/home/athenes/Paraplume/benchmark/mipe/3D/
pdb_folder_path=/home/athenes/all_structures/imgt_renumbered_mipe
cvs=("0" "1" "2" "3" "4")

for seed in "${seeds[@]}" ; do
    for cv in "${cvs[@]}" ; do
        python graph_extension/train_graph_light.py ${train_folder}_${cv} ${val_folder}_${cv} ${train}_${cv}.csv ${val}_${cv}.csv \
        --seed $seed --patience 300 -n 300 -r ${result_folder}seed${seed}/$cv --num-workers 16\
        --gpu 1 --pdb-folder-path-train $pdb_folder_path \
        --pdb-folder-path-val $pdb_folder_path
        python graph_extension/evaluate_graph.py ${result_folder}seed${seed}/$cv $test_folder $test_csv --pdb-folder-path-test $pdb_folder_path
        python graph_extension/predict_graph.py $result_folder/seed$seed/$cv/graph_checkpoint.pt $test_folder $test_csv --pdb-folder-path $pdb_folder_path
        python graph_extension/predict_graph.py $result_folder/seed$seed/$cv/graph_checkpoint.pt $test_folder $test_csv --pdb-folder-path /home/athenes/all_structures/abb3_mipe_renumbered --name abb3
    done
done
