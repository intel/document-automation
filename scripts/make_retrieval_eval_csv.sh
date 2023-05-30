#!/bin/bash

DEVJSON="/home/user/docvqa/docvqa_dev.json"
SAVEPATH="/home/user/output/processed_data/docvqa_dev.csv"

python src/convert_dev_json_to_csv.py --json_file ${DEVJSON} --save_to ${SAVEPATH}