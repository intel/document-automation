#!/bin/bash

echo "stop and clean up haystack and database containers"
docker stop $(docker ps -a |grep -E 'haystack|head|ray|elasticsearch|postgres'|awk '{print $1 }')
docker rm $(docker ps -a |grep -E 'haystack|head|ray|elasticsearch|postgres'|awk '{print $1 }')