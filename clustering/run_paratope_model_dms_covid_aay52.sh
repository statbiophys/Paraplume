#!/bin/bash

# Define the base paths
input_dir="/home/athenes/paratope_clustering/dms_covid"
output_dir="/home/athenes/benchmark2/paragraph_all/results"
script="python predict_from_seq.py predict-from-df"

prefix="aay52"
i=0  # Start with file suffix 0
while true; do
    input_file="${input_dir}/${prefix}/${prefix}_${i}.csv"
    output_file="${input_dir}/${prefix}/paratope_${prefix}_${i}.csv"  # Define the expected output file name

    if [[ -f "$input_file" ]]; then
        if [[ -f "$output_file" ]]; then
            echo "Output file $output_file already exists. Skipping $input_file."
        else
            echo "Processing $input_file"
            $script "$input_file" "$output_dir"
        fi
    else
        echo "File $input_file does not exist. Moving to the next prefix."
        break
    fi
    ((i++))
done
