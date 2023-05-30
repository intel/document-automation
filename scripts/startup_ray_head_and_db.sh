#!/bin/bash

head_ip=$HEAD_IP

echo "Head node IP is "$head_ip
echo "http proxy on your machine is "$http_proxy
echo "https proxy on your machine is "$https_proxy

# setting threads
num_threads_db=8 # num of threads to be used for databases
# num_threads to be used for indexing
total_threads=`cat /proc/cpuinfo |grep physical\ id|wc -l`
num_threads_indexing=$(expr $total_threads - $num_threads_db) 

# directories
dataset=$(dirname $PWD)/dataset/
source_code=$PWD #NFS, root dir of use case repo
output=$(dirname $PWD)/output #NFS, folder that will store dpr models and faiss index files 
docvqa=$(dirname $PWD)/dureader_vis_docvqa # need to mount this folder to get access to dev file for retrieval eval

echo $dataset
echo $source_code
echo $docvqa
echo $output

elasticsearch_data=$DB_DIR/esdb
postgres_data=$DB_DIR/db

echo "elasticsearch database directory is "$elasticsearch_data
echo "postgresql database directory is "$postgres_data


database=all #options = [elasticsearch, postgresql,all].


echo "Starting up containers on head node..."
bash scripts/run-ray-cluster.sh -r startup_all -e $num_threads_db -c $num_threads_indexing -u $head_ip -d $database -f $dataset  -w $source_code -b $elasticsearch_data -p $postgres_data -q $output -n $docvqa

# prepare head node haystack-api container
echo "Preparing head-node haystack-api container, this may take a few minutes..."
docker exec -i head-indexing /bin/bash -c "cd /home/user/application; bash scripts/prepare_env_indexing.sh; bash scripts/make_retrieval_eval_csv.sh"

echo "Completed prep on head node!"


echo "Going inside haystack-api container on head node..."
docker exec -it head-indexing /bin/bash