cvs=("0" "1" "2" "3" "4")
for cv in "${cvs[@]}" ; do
    python paraplume/create_dataset.py /home/athenes/Paraplume/datasets/mipe/train_$cv.csv -r /home/athenes/Paraplume/benchmark/mipe --pdb-folder-path /home/athenes/all_structures/imgt_renumbered_mipe
    python paraplume/create_dataset.py /home/athenes/Paraplume/datasets/mipe/val_$cv.csv -r /home/athenes/Paraplume/benchmark/mipe --pdb-folder-path /home/athenes/all_structures/imgt_renumbered_mipe
done
python paraplume/create_dataset.py /home/athenes/Paraplume/datasets/mipe/test_set.csv -r /home/athenes/Paraplume/benchmark/mipe --pdb-folder-path /home/athenes/all_structures/imgt_renumbered_mipe
