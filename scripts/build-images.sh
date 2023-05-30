#!/bin/bash
# You can remove build-arg http_proxy and https_proxy if your network doesn't need it
# no_proxy="localhost,127.0.0.0/1"
# proxy_server="" # your http proxy server
http_proxy_server="http://fmproxyslb.ice.intel.com:911"
https_proxy_server="http://fmproxyslb.ice.intel.com:911"
#dockerfile=/localdisk/minminho/applications.ai.appliedml.workflow.odqa/docker/Dockerfile-debug

echo -e "\nbuild haystack-api image\n"

cd ../applications.ai.appliedml.workflow.odqa/applications/indexing/
echo $PWD

DOCKER_BUILDKIT=0 docker build \
    -f ../../docker/Dockerfile ../../ \
    -t intel/ai-workflows:odqa-haystack-api\
    --network=host \
    --build-arg http_proxy=${http_proxy_server} \
    --build-arg https_proxy=${https_proxy_server} \
    --build-arg no_proxy=${no_proxy}
    
echo -e "\nbuild haystack-ui image\n"

cd ../../ui
echo $PWD

DOCKER_BUILDKIT=0 docker build -t intel/ai-workflows:odqa-haystack-ui . \
    --network=host \
    --build-arg http_proxy=${http_proxy_server} \
    --build-arg https_proxy=${https_proxy_server} \
    --build-arg no_proxy=${no_proxy}
    