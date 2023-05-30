#!/bin/bash

# bm25
# config=docker/env.bm25
# yaml_file=docker/docker-compose-bm25.yml

# dpr
# config=docker/env.dpr
# yaml_file=docker/docker-compose-dpr.yml

# ensemble
config=docker/env.ensemble
yaml_file=docker/docker-compose-ensemble.yml

docker compose --env-file $config -f $yaml_file up #--build