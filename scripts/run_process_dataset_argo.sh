#!/bin/bash


METHOD="v0" #v0-use docvqa_train.json text, v1-use our own ocr pipeline

MAX_LEN_PASSAGE=500
OVERLAP=10
MINCHARS=5

# directories
DATAPATH="/home/user/docvqa/" 
TRAIN="docvqa_train.json"
DEV="docvqa_dev.json"
SAVEPATH="/home/user/output/processed_data/"
[ -d $SAVEPATH ] || mkdir -p $SAVEPATH

echo "Processing data, this may take a while...."
python src/process_dataset.py --method ${METHOD} --max_seq_len_passage ${MAX_LEN_PASSAGE} --overlap ${OVERLAP} --min_chars ${MINCHARS} --data_dir ${DATAPATH} --train_file ${TRAIN} --save_to ${SAVEPATH} --dev_file ${DEV} --process_dev


