#!/bin/bash

RETRIEVAL_METHOD="all" # options: bm25, dpr, all

POSTGRES_IP=${POSTGRES_IP:-${HEAD_IP}}
DB="postgresql://postgres:postgres@"$POSTGRES_IP":5432/haystack"
echo $DB
ESDB_IP=${ESDB_IP:-${HEAD_IP}}
echo $ESDB_IP

# image preprocessing args
PREPROCESS="grayscale" #options: grayscale, binarize, none
# You can append --crop_image to the python script command to crop images

# OCR engine
OCR="paddleocr" # options: tesseract, paddleocr

# params for post processing ocr outputs into passages
MAX_LEN_PASSAGE=500
OVERLAP=10
MINCHARS=5 
# You can append --split_doc or not to the python script command to split text into passages or not


# DPR Encoders used to embed passages
MODEL_NAME=${MODEL_NAME:-"my_dpr_model"}
QUERYENCODER="/home/user/output/dpr_models/${MODEL_NAME}/query_encoder"
DOCENCODER="/home/user/output/dpr_models/${MODEL_NAME}/passage_encoder"
[ -d $QUERYENCODER ] || mkdir -p $QUERYENCODER
[ -d $DOCENCODER ] || mkdir -p $DOCENCODER
# indexing output
INDEX_DIR="/home/user/output/index_files"
[ -d $INDEX_DIR ] || mkdir -p $INDEX_DIR
INDEXFILE="/home/user/output/index_files/faiss-indexfile.faiss"
INDEXJSON="/home/user/output/index_files/faiss-indexfile.json"
if [ -f "$INDEXJSON" ]; then
  sed -i "s/\([0-9]\{1,3\}\.\)\{3\}[0-9]\{1,3\}/$POSTGRES_IP/g" $INDEXJSON
fi
INDEXNAME="dureadervis-documents"

# Ray params
RAY_WRITING_BS=10000
RAY_EMBED_BS=50
RAY_PREPROCESS_MIN_ACTORS=8
RAY_PREPROCESS_MAX_ACTORS=20
RAY_EMBED_MIN_ACTORS=4
RAY_EMBED_MAX_ACTORS=8
RAY_PREPROCESS_CPUS_PER_ACTOR=4
RAY_WRITING_CPUS_PER_ACTOR=4
RAY_EMBED_CPUS_PER_ACTOR=20


python src/gen-sods-doc-image-ray.py --retrieval_method ${RETRIEVAL_METHOD} --db ${DB} --esdb ${ESDB_IP} --preprocess ${PREPROCESS} --ocr_engine ${OCR} --max_seq_len_passage ${MAX_LEN_PASSAGE} --overlap ${OVERLAP} --min_chars ${MINCHARS} --query_encoder ${QUERYENCODER} --doc_encoder ${DOCENCODER} --index_file ${INDEXFILE} --index_name ${INDEXNAME} --writing_bs ${RAY_WRITING_BS} --embedding_bs ${RAY_EMBED_BS} --preprocess_min_actors ${RAY_PREPROCESS_MIN_ACTORS} --preprocess_max_actors ${RAY_PREPROCESS_MAX_ACTORS} --embedding_min_actors ${RAY_EMBED_MIN_ACTORS} --embedding_max_actors ${RAY_EMBED_MAX_ACTORS} --preprocess_cpus_per_actor ${RAY_PREPROCESS_CPUS_PER_ACTOR} --writing_cpus_per_actor ${RAY_WRITING_CPUS_PER_ACTOR} --embedding_cpus_per_actor ${RAY_EMBED_CPUS_PER_ACTOR} --split_doc --add_doc --embed_doc #--toy_example