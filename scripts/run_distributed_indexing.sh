#!/bin/bash

RETRIEVAL_METHOD="all" # options: bm25, dpr, all

DB="postgresql://postgres:postgres@"$HEAD_IP":5432/haystack"
echo $DB
ESDB=${HEAD_IP}
echo $ESDB

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
QUERYENCODER="/home/user/output/dpr_models/my_dpr_model/query_encoder" 
DOCENCODER="/home/user/output/dpr_models/my_dpr_model/passage_encoder"

# indexing output
INDEXFILE="/home/user/output/index_files/faiss-indexfile.faiss"
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


python src/gen-sods-doc-image-ray.py --retrieval_method ${RETRIEVAL_METHOD} --db ${DB} --esdb ${ESDB} --preprocess ${PREPROCESS} --ocr_engine ${OCR} --max_seq_len_passage ${MAX_LEN_PASSAGE} --overlap ${OVERLAP} --min_chars ${MINCHARS} --query_encoder ${QUERYENCODER} --doc_encoder ${DOCENCODER} --index_file ${INDEXFILE} --index_name ${INDEXNAME} --writing_bs ${RAY_WRITING_BS} --embedding_bs ${RAY_EMBED_BS} --preprocess_min_actors ${RAY_PREPROCESS_MIN_ACTORS} --preprocess_max_actors ${RAY_PREPROCESS_MAX_ACTORS} --embedding_min_actors ${RAY_EMBED_MIN_ACTORS} --embedding_max_actors ${RAY_EMBED_MAX_ACTORS} --preprocess_cpus_per_actor ${RAY_PREPROCESS_CPUS_PER_ACTOR} --writing_cpus_per_actor ${RAY_WRITING_CPUS_PER_ACTOR} --embedding_cpus_per_actor ${RAY_EMBED_CPUS_PER_ACTOR} --split_doc --add_doc --embed_doc #--toy_example