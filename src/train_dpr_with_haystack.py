# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
from haystack.nodes.retriever.dense import DensePassageRetriever
import argparse


def parse_cmd():
    desc = 'fine tuning DPR models...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    
    args.add_argument('--data_dir', type=str, default='/home/user/output/processed_data/', dest='data_dir')
    args.add_argument('--train_file', type=str, default='processed_train.json', dest='train_file')
    args.add_argument('--dev_file', type=str, default='processed_dev.json', dest='dev_file')
    args.add_argument('--save_to', type=str, default='/home/user/output/dpr_models/', dest='save_to', help='directory to save fine tuned models')
    
    # training hyperparams
    args.add_argument('--num_hard_neg', type=int, default=0, dest='num_hard_neg', help='number of hard negatives per positive sample')
    args.add_argument('--bs', type=int, default=128, dest='bs', help='batch size in training')
    args.add_argument('--epochs', type=int, default=3, dest='epochs', help='number of epochs in training')
    args.add_argument('--eval_every', type=int, default=87, dest='eval_every', help='evaluate on dev set every specified number of training steps')
    args.add_argument('--lr', type=float, default=1e-5, dest='lr', help='learning rate')
    args.add_argument('--warmup', type=int, default=20, dest='warmup', help='number of warmup steps in training')
    
    # model params
    args.add_argument('--query_encoder', type=str, default='microsoft/infoxlm-base', dest='query_encoder', help='pretrained model name on HF model hub for query encoder')
    args.add_argument('--doc_encoder', type=str, default='microsoft/infoxlm-base', dest='doc_encoder', help='pretrained model name on HF model hub for document encoder')
    args.add_argument('--max_len_query', type=int, default=64, dest='max_len_query', help='max sequence length for query')
    args.add_argument('--max_len_passage', type=int, default=500, dest='max_len_passage', help='max sequence length for document passage')
 
    print(args.parse_args())
    return args.parse_args()
    

    
def train_dpr(cfg):
    print('Initiate DPR models...')
    retriever = DensePassageRetriever(document_store=None,
                                                  query_embedding_model=cfg.query_encoder,
                                                  passage_embedding_model=cfg.doc_encoder,
                                                  max_seq_len_query=cfg.max_len_query,
                                                  max_seq_len_passage=cfg.max_len_passage,
                                                  batch_size=cfg.bs,
                                                  use_gpu=False,
                                                  #embed_title=True,
                                                  xlm_roberta=True,
                                                  use_fast_tokenizers=True)


    print('Start fine tuning...')
    retriever.train(
            data_dir = cfg.data_dir,
            train_filename = cfg.train_file,
            dev_filename = cfg.dev_file,
            test_filename = None,
            #max_samples: Optional[int] = None,
            #max_processes: int = 128,
            #multiprocessing_strategy: Optional[str] = None,
            dev_split = 0,
            batch_size= cfg.bs,
            embed_title = False,
            num_hard_negatives= cfg.num_hard_neg,
            num_positives = 1,
            n_epochs = cfg.epochs,
            evaluate_every = cfg.eval_every,
            n_gpu = 0,
            learning_rate = cfg.lr,
            epsilon = 1e-08,
            weight_decay = 0.0,
            num_warmup_steps= cfg.warmup,
            grad_acc_steps = 1,
            use_amp = False,
            optimizer_name = "AdamW",
            optimizer_correct_bias = True,
            save_dir = cfg.save_to,
            query_encoder_save_dir = "query_encoder",
            passage_encoder_save_dir = "passage_encoder",
            #checkpoint_root_dir: Path = Path("model_checkpoints"),
            #checkpoint_every: Optional[int] = None,
            #checkpoints_to_keep: int = 3,
            #early_stopping: Optional[EarlyStopping] = None,
        )
          
    print('Completed fine tuning!')
    

    
if __name__ == "__main__":
    
    cfg = parse_cmd()
    train_dpr(cfg)
    


