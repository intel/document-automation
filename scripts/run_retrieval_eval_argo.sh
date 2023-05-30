#!/bin/bash

# eval data
DATAPATH="/home/user/output/processed_data/docvqa_dev.csv"

# retrieval params
RETRIEVAL_METHOD="ensemble" #options: bm25, dpr, ensemble
TOPK=100
RERANK_TOPK=5
WEIGHT=1.5

# elasticsearch database
ESDB_IP=${ESDB_IP:-${HEAD_IP}}
PORT=9200

# FAISS index file and index name
INDEXFILE="/home/user/output/index_files/faiss-indexfile.faiss"
INDEXJSON="/home/user/output/index_files/faiss-indexfile.json"
if  [ -f "$INDEXJSON" ] && [ -n "$POSTGRES_IP" ]; then
  sed -i "s/\([0-9]\{1,3\}\.\)\{3\}[0-9]\{1,3\}/$POSTGRES_IP/g" $INDEXJSON
fi
INDEXNAME="dureadervis-documents"


# params for dpr retriever
MODEL_NAME=${MODEL_NAME:-"my_dpr_model"}
QUERYENCODER="/home/user/output/dpr_models/${MODEL_NAME}/query_encoder"
DOCENCODER="/home/user/output/dpr_models/${MODEL_NAME}/passage_encoder"
[ -d $QUERYENCODER ] || mkdir -p $QUERYENCODER
[ -d $DOCENCODER ] || mkdir -p $DOCENCODER
MAX_LEN_PASSAGE=500
MAX_LEN_QUERY=64


echo "Evaluating retrieval performance..."

python src/test_retrieval_pipeline.py --topk ${TOPK} --retrieval_method ${RETRIEVAL_METHOD} --datapath ${DATAPATH} --index_name ${INDEXNAME} --host ${ESDB_IP} --port ${PORT}  --index_file ${INDEXFILE} --query_encoder ${QUERYENCODER} --doc_encoder ${DOCENCODER} --max_seq_len_passage ${MAX_LEN_PASSAGE} --max_seq_len_query ${MAX_LEN_QUERY} --rerank_topk ${RERANK_TOPK} --weight ${WEIGHT}
