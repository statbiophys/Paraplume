#!/bin/bash

# Output file for timing and RAM usage
output_file="timing_results_abb3_cpu.txt"
echo "ModelType,Seed,Size,TimeTaken(s)" > $output_file

for seed in 0; do
    for size in 0 10 100 1000; do
        echo "ABB3, Seed $seed, Size $size..."
        start_time=$(date +%s)
        python /home/athenes/Paraplume/data_with_scripts/speed_tests/compute_3D_structures.py /home/athenes/Paraplume/data_with_scripts/speed_tests/test$size.csv --gpu -1
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "ABB3,$seed,$size,$duration" >> $output_file
    done
done
