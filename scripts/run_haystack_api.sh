#!/bin/bash


INDEXJSON="/home/user/output/index_files/faiss-indexfile.json"
if [ -f "$INDEXJSON" ]; then
  sed -i "s/\([0-9]\{1,3\}\.\)\{3\}[0-9]\{1,3\}/$POSTGRES_IP/g" $INDEXJSON
fi

RETRIEVAL_METHOD=${RETRIEVAL_METHOD:-"ensemble"}
sed "s/localhost/$DOCUMENTSTORE_PARAMS_HOST/g" /home/user/application/configs/pipelines_${RETRIEVAL_METHOD}.haystack-pipeline.yml > $PIPELINE_YAML_PATH


gunicorn rest_api.application:app -b 0.0.0.0 -k uvicorn.workers.UvicornWorker --workers 1 --timeout 180