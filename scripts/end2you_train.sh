#!/bin/bash

# activate environment
source ./activate end2you

root_dir="./dist/"
save_path="./end2you_files"

# Start training
python -m src.end2you --modality="audio" \
               --root_dir=end2you_files/training \
               --batch_size=8 \
               --model_name=emo18 \
               --num_outputs=3 \
               train  \
               --loss=ce \
               --metric=uar \
               --train_dataset_path=$save_path/data/train \
               --valid_dataset_path=$save_path/data/devel \
               --num_epochs=30 \
               --learning_rate=0.001

# Start evaluation
python -m src.end2you --modality="audio" \
               --root_dir=$save_path/ \
               --model_name=emo18 \
               --num_outputs=3 \
               test  \
               --prediction_file=$save_path/training/predictions.csv \
               --metric=uar \
               --dataset_path=$save_path/data/test \
               --model_path=$save_path/training/model/best.pth.tar