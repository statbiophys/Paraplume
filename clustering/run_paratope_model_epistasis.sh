#!/bin/bash

# Define the base paths
input_dir="/home/athenes/paratope_clustering/epistasis/splits"
output_dir="/home/athenes/benchmark2/paragraph_all/results"
script="python predict_from_seq.py predict-from-df"

prefix="9114_sequences"
i=0  # Start with file suffix 0
while true; do
    input_file="${input_dir}/${prefix}_${i}.csv"
    if [[ -f "$input_file" ]]; then
        echo "Processing $input_file"
        $script "$input_file" "$output_dir"
    else
        echo "File $input_file does not exist. Moving to the next prefix."
        break
    fi
    ((i++))
done
