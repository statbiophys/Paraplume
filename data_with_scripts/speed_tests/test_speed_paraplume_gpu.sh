#!/bin/bash

# Output file for timing and RAM usage
output_file="timing_results_paraplume_gpu.txt"
echo "ModelType,Seed,Size,TimeTaken(s)" > $output_file

for seed in 0 1 2; do
    for size in 1 10 100 1000 10000; do
        echo "Small model, Seed $seed, Size $size..."
        start_time=$(date +%s)
        python /home/athenes/Paraplume/paraplume/infer.py file-to-paratope \
            /home/athenes/Paraplume/data_with_scripts/speed_tests/test$size.csv \
            --emb-proc-size 256 --small

        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Small,$seed,$size,$duration" >> $output_file
    done

    for size in 1 10 100 1000 10000; do
        echo "Large model, Seed $seed, Size $size..."
        start_time=$(date +%s)
        python /home/athenes/Paraplume/paraplume/infer.py file-to-paratope \
            /home/athenes/Paraplume/data_with_scripts/speed_tests/test$size.csv \
            --emb-proc-size 256
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Large,$seed,$size,$duration" >> $output_file
    done
done
