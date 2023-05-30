#!/bin/bash

# eval data
DATAPATH="/home/user/output/processed_data/docvqa_dev.csv"

# retrieval params
RETRIEVAL_METHOD="ensemble" #options: bm25, dpr, ensemble
TOPK=100
RERANK_TOPK=10
WEIGHT=1.5

# elasticsearch database
HOST="localhost"
PORT=9200

# FAISS index file and index name
INDEXFILE="/home/user/output/index_files/faiss-indexfile.faiss"
INDEXNAME="dureadervis-documents"


# params for dpr retriever
QUERYENCODER="/home/user/output/dpr_models/my_dpr_model/query_encoder" 
DOCENCODER="/home/user/output/dpr_models/my_dpr_model/passage_encoder"
MAX_LEN_PASSAGE=500
MAX_LEN_QUERY=64


echo "Evaluating retrieval performance..."

python src/test_retrieval_pipeline.py --topk ${TOPK} --retrieval_method ${RETRIEVAL_METHOD} --datapath ${DATAPATH} --index_name ${INDEXNAME} --host ${HOST} --port ${PORT}  --index_file ${INDEXFILE} --query_encoder ${QUERYENCODER} --doc_encoder ${DOCENCODER} --max_seq_len_passage ${MAX_LEN_PASSAGE} --max_seq_len_query ${MAX_LEN_QUERY} --rerank_topk ${RERANK_TOPK} --weight ${WEIGHT}
