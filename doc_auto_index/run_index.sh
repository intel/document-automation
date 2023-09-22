#!/bin/bash
ARGS=`getopt -a -o D:O:Q:d:P -l dataset_path:,output_path:,query_encoder:,doc_encoder:,paddleocr_path: -- "$@"`
# function usage() {
#     echo  'help'
# }
# [ $? -ne 0 ] && usage
#set -- "${ARGS}"
eval set -- "${ARGS}"
while true
do
    case "$1" in
        -D|--dataset_path)
            dataset_path="$2"
            shift
            ;;
        -Q|--query_encoder)
            query_encoder="$2"
            shift
            ;;
        -d|--doc_encoder)
            doc_encoder="$2"
            shift
            ;;
        -O|--output_path)
            output_path="$2"
            shift
            ;;
        -P|--paddleocr_path)
            paddleocr_path="$2"
            shift
            ;;
        --)
            shift;
            break;;
    esac
shift
done

if [ -z "$query_encoder" ]; then
    echo "Argument --query_encoder is empty or not provided"
    exit 1
fi

if [ -z "$doc_encoder" ]; then
    echo "Argument --doc_encoder is empty or not provided"
    exit 1
fi

if [ -z "$dataset_path" ]; then
    echo "Argument --dataset_path is empty or not provided"
    exit 1
fi

export WORKSPACE=$(realpath .)
export OUTPUT_PATH=${output_path:-$WORKSPACE/output}
export DB_DIR=$OUTPUT_PATH/databases

echo "The output path is ${OUTPUT_PATH}, and the DB_DIR is ${DB_DIR}."

mkdir -p $WORKSPACE/dataset $OUTPUT_PATH
mkdir -p $OUTPUT_PATH/index_files
mkdir -p $DB_DIR/esdb
mkdir -p $DB_DIR/db

apt-get update

# install PostgreSQL
apt-get install -y postgresql

# install Elasticsearch
apt-get install -y apt-transport-https
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add -
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | tee /etc/apt/sources.list.d/elastic-7.x.list
apt-get update && apt-get install -y elasticsearch

# configure Elasticsearch
chown -R elasticsearch:elasticsearch $DB_DIR/esdb
sed -i "s|path.data: /var/lib/elasticsearch|path.data: $DB_DIR/esdb|" /etc/elasticsearch/elasticsearch.yml
echo 'discovery.type: single-node' >> /etc/elasticsearch/elasticsearch.yml

# launch Elasticsearch
service elasticsearch start

echo "Configuring PostgreSQL..."
service postgresql stop

# move data directory to output path
NEW_PG_DATA="$DB_DIR/db"
mv /var/lib/postgresql/12/main $NEW_PG_DATA

chown -R postgres:postgres $NEW_PG_DATA

sed -i "s|data_directory = '/var/lib/postgresql/12/main'|data_directory = '$NEW_PG_DATA/main'|g" /etc/postgresql/12/main/postgresql.conf

service postgresql start

su - postgres -c "psql -c \"CREATE DATABASE haystack;\""
su - postgres -c "psql -c \"ALTER USER postgres WITH PASSWORD 'postgres'\""

echo "PostgreSQL is ready."
apt-get clean
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"
ray start --node-ip-address='127.0.0.1' --head --dashboard-host='0.0.0.0' --dashboard-port=8265 --disable-usage-stats && \
(python test_pocr.py || (mkdir -p /root/.paddleocr/whl && cp -r $paddleocr_path/* /root/.paddleocr/whl/)) && \
python doc_auto_index.py --retrieval_method all --db postgresql://postgres:postgres@localhost:5432/haystack \
--esdb localhost --preprocess grayscale --ocr_engine paddleocr --max_seq_len_passage 500 --overlap 10 --min_chars 5 \
--query_encoder $query_encoder --doc_encoder $doc_encoder \
--index_file $OUTPUT_PATH/index_files/faiss-indexfile.faiss --index_name dureadervis-documents --writing_bs 10000 --embedding_bs 50 \
--preprocess_min_actors 8 --preprocess_max_actors 20 --embedding_min_actors 4 --embedding_max_actors 8 --preprocess_cpus_per_actor 4 \
--writing_cpus_per_actor 4 --embedding_cpus_per_actor 20 --split_doc --add_doc --embed_doc --dataset_path $dataset_path #--toy_example
