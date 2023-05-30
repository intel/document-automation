# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
import pandas as pd
import time
import argparse

def parse_cmd():
    desc = 'Make dev csv file for retrieval eval...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('--json_file', type=str, default='/home/user/dataset/docvqa_dev.json', dest='json_file')
    args.add_argument('--save_to', type=str, default='/home/user/output/processed_data/docvqa_dev.csv', dest='save_to', help='directory to save csv file')
    print(args.parse_args())
    return args.parse_args()
    
if __name__ == "__main__":
    
    cfg = parse_cmd()  

    print('Converting {} to csv....'.format(cfg.json_file))
    t0 = time.time()
    dev_file = ''

    df = pd.read_json(cfg.json_file, lines=True)

    df.to_csv(cfg.save_to)
    t1=time.time()
    print('Completed conversion!')
    print('Time to convert: {:.2f} sec'.format(t1-t0))