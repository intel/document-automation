#!/bin/bash

# dataset download directory
DATASET_PATH="/home/user/dataset/" 

echo "Download dataset, this may take a while...."
python src/download_dataset.py --dataset_dir ${DATASET_PATH}