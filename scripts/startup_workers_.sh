#!/bin/bash

head_ip=$HEAD_IP
echo "head node ip is "$head_ip
echo "http proxy on your machine is "$http_proxy
echo "https proxy on your machine is "$https_proxy

num_threads=96

dataset=$(dirname $PWD)/dataset/
source_code=$PWD #NFS, root dir of use case repo
output=$(dirname $PWD)/output #NFS, folder that will store dpr models and faiss index files 

echo $dataset
echo $source_code
echo $output

# one worker per node to maximize performance
bash scripts/run-ray-cluster.sh -r startup_workers -s $num_threads -u $head_ip  -f $dataset -w $source_code -q $output

echo "Preparing worker-node haystack-api container environment..."
docker exec -i ray-0-indexing /bin/bash -c "cd /home/user/application; bash scripts/prepare_env_indexing.sh"

echo "Completed prep on worker node!"