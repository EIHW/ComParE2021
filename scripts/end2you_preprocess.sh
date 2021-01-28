#!/bin/bash

# activate environment
source ./activate end2you

root_dir="./dist/"
save_path="./end2you_files"

python src/convert_to_e2u.py --covid_path=$root_dir --save_path=$save_path

partitions=("train" "devel" "test")


mkdir -p end2you_files

# Start generating hdf5 data for all partitions
for p in ${partitions[@]}; do
    python -m src.end2you --modality="audio" \
                   --root_dir=$save_path/data \
                   generate \
                   --input_file=$save_path/labels/"$p"_input_file.csv \
                   --save_data_folder=$save_path/data/$p
done

rm -Rf $save_path/labels