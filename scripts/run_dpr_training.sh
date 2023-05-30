#!/bin/bash

# directories
MODEL_NAME=${MODEL_NAME}
echo "MODEL NAME: $MODEL_NAME"
DATAPATH="/home/user/output/processed_data/" 
TRAIN="processed_train.json"
DEV="processed_dev.json"
SAVEPATH="/home/user/output/dpr_models/${MODEL_NAME}"
echo $SAVEPATH
  
    
# training hyperparams   
BATCHSIZE=128
EPOCHS=3
LR=1e-5
WARMUP=20
EVAL_EVERY=87
NUM_HARD_NEG=0

# model params
QUERY_ENCODER="microsoft/infoxlm-base"
DOC_ENCODER="microsoft/infoxlm-base"
MAX_LEN_QUERY=64
MAX_LEN_PASSAGE=500


echo "Fine tuning DPR models, this may take a while..."
python src/train_dpr_with_haystack.py --data_dir ${DATAPATH} --train_file ${TRAIN} --dev_file ${DEV} --save_to ${SAVEPATH} --query_encoder ${QUERY_ENCODER} --doc_encoder ${DOC_ENCODER} --max_len_query ${MAX_LEN_QUERY} --max_len_passage ${MAX_LEN_PASSAGE} --num_hard_neg ${NUM_HARD_NEG} --bs ${BATCHSIZE} --epochs ${EPOCHS} --lr ${LR} --warmup ${WARMUP} --eval_every ${EVAL_EVERY}











