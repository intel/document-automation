#!/bin/bash

# set dataset download directory
DATASET_DIR=$NFS_DIR/dataset

# download dureader dataset
yes | conda create --force -n dureader_downloader python=3.9
conda activate dureader_downloader
echo "Download dataset, this may take a while...."
python -m pip install dataset-librarian
[[ -d ${DATASET_DIR} ]] || mkdir -p ${DATASET_DIR}
rm -rf ${DATASET_DIR}/*
python src/download_dataset.py --dataset_dir ${DATASET_DIR}
conda deactivate

# post jobs
cd $DATASET_DIR
tar -xzf dureader_vis_images_part_1.tar.gz
tar -xzf dureader_vis_images_part_2.tar.gz
tar -xzf dureader_vis_images_part_3.tar.gz
tar -xzf dureader_vis_images_part_4.tar.gz
tar -xzf dureader_vis_images_part_5.tar.gz
tar -xzf dureader_vis_images_part_6.tar.gz
tar -xzf dureader_vis_images_part_7.tar.gz
tar -xzf dureader_vis_images_part_8.tar.gz
tar -xzf dureader_vis_images_part_9.tar.gz
tar -xzf dureader_vis_images_part_10.tar.gz
mv dureader_vis_docvqa.tar.gz ../
cd ..
tar -xzf dureader_vis_docvqa.tar.gz