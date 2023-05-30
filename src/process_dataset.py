# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
import json
from utils import get_split
from utils import preprocess_single_image, apply_tesseract, postprocess_ocr_outputs_of_single_image
from utils import get_all_image_path

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever
from haystack import Pipeline
from haystack.pipelines import DocumentSearchPipeline

from tqdm.auto import tqdm
import argparse
import random

class TextProcessConfig():
    def __init__(self, config_dict):
        self.max_seq_len_passage = config_dict['max_seq_len_passage']
        self.overlap = config_dict['overlap']
        self.min_chars = config_dict['min_chars']


def find_positive_passage(doc_text, answer, cfg, split_into_passages=True): 
    # split doc into passages of max_seq_len with overlap
    if split_into_passages == True:
        passages = get_split(doc_text, cfg)
    else:
        passages = [d['content'] for i, d in enumerate(doc_text)]
    
    #print('# of passages:', len(passages))
    #print('passages:\n')
    #print(passages)
    
    if len(passages) == 1:
        #print('short text, return the entire text as positive passage')
        return passages[0]
    
    max_overlap = 0
    pos_passage = ''
    for p in passages:
        if answer in p:
            # if a passage contains answer, then get that passage
            #print('found answer in passage')
            return p
        # find passage that has the highest overlap with answer
        #print('search for max overlap')
        text_overlap = len(set(p).intersection(set(answer)))
        #print('passages: ', p)
        #print('overlap: ', text_overlap)
        if text_overlap > max_overlap:
            pos_passage = p
            max_overlap = text_overlap
    return pos_passage


def process_dataset_for_dpr(cfg, data_path, save_path, folder_nums=None, tokenizer=None, model=None):
    # method_version: str - v0, v1
    # v0: use Dureader-vis Paddle OCR results for training
    # v1: process image with our own indexing method to get positive passage
    if cfg.hard_neg == True:
        p_retrieval = config_retrieval_pipeline(cfg)
    
    print('Reading data...')    
    with open(data_path) as json_file:
        lines = json_file.readlines()
    print('Read complete!')
    
    if cfg.method == 'v1':
        full_path_all, img_id_all = get_all_image_path(folder_nums, cfg.folder_prefix)
        
    
    samples = []
    
    print('Start processing data....')
    progress_bar = tqdm(range(len(lines)))
    for line in lines:
        parsed=json.loads(line)
        question = parsed['question']
        #print('question: ', question)
        answer = ''.join(parsed['answer'])
        #print('answer: ', answer)
        if cfg.method == 'v0': # use baidu provided ocr results
            doc = ''.join(parsed['document'])
            pos_passage=find_positive_passage(doc, answer, cfg, split_into_passages=True)
     
        elif cfg.method == 'v1': #use our own ocr pipeline
            image_id = parsed['image_id']
            if image_id in img_id_all:
                image_path = full_path_all[img_id_all.index(image_id)]
                try:
                    doc = extract_text_from_image(image_path, cfg, tokenizer=tokenizer, model=model)
                    pos_passage=find_positive_passage(doc, answer, cfg, split_into_passages=False)
                except:
                    pos_passage=None
            else:
                pos_passage = None
                #print('answer image not in scope')
                continue
        else:
            raise NotImplementedError
        
        if pos_passage!=None and cfg.hard_neg==True:
            hard_neg_passage = get_hard_negatives(question, p_retrieval, cfg.num_retrieve, 1, image_id)[0]
            
            sample = {
                    'question': question,
                    'answers': answer,
                    'positive_ctxs': [{'title':'','text':pos_passage, 'passage_id':''}],
                    "negative_ctxs": [],
                    "hard_negative_ctxs": [{'title':'','text':hard_neg_passage, 'passage_id':''}],
                }
            samples.append(sample)
            
        elif pos_passage!=None and cfg.hard_neg==False:
            sample = {
                    'question': question,
                    'answers': answer,
                    'positive_ctxs': [{'title':'','text':pos_passage, 'passage_id':''}],
                    "negative_ctxs": [],
                    "hard_negative_ctxs": [],
                }
            samples.append(sample)
            
        else:
            pass
        
        
            
        progress_bar.update(1)

        
    print('Completed processing data!')
    
    print('Saving processed data...')
    with open(save_path, "w", encoding="utf-8") as json_ds:
        json.dump(samples, json_ds, indent=4, ensure_ascii=False)
    print('Save complete!')

    

def process_dataset_for_cross_encoder(cfg):
    if cfg.hard_neg == True:
        p_retrieval = config_retrieval_pipeline(cfg)
        
    with open(cfg.data_path) as json_file:
        lines = json_file.readlines()

    pos_samples = []
    queries = []
    samples = []
    correct_image_id = []
    
    progress_bar = tqdm(range(len(lines)))
    print('Get positive passages...')
    for line in lines:
        parsed=json.loads(line)
        question = parsed['question']
        queries.append(question)
        
        correct_image_id.append(parsed['image_id'])
        
        #print('question: ', question)
        answer = ''.join(parsed['answer'])
        #print('answer: ', answer)
        doc = ''.join(parsed['document'])
        pos_passage=find_positive_passage(doc, answer, cfg)
        pos_samples.append(pos_passage)
        
        
        if cfg.process_dev == False:
            sample = {
                    'question': question,
                    'passage': pos_passage,
                    'label':1
                }

            samples.append(sample)
        progress_bar.update(1)

    assert len(queries) == len(pos_samples), "num of queries not equal to num of pos samples"
    #print('# of positive passages:', len(pos_samples))
    
    # sample negatives
    print('Get negative passages...')
    progress_bar2 = tqdm(range(len(queries))) 
    neg_ratio = cfg.neg_ratio
    for i in range(len(queries)):
        if cfg.hard_neg:
            negs = get_hard_negatives(queries[i], p_retrieval, 200, neg_ratio, correct_image_id[i])
        else:
            pos = pos_samples[i]
            neg_candidates = pos_samples[:i] + pos_samples[i+1:]
            #print('# of neg candidates: ', len(neg_candidates))
            # ranomly sample from neg candidates
            negs = random.sample(neg_candidates, neg_ratio)
        
        if cfg.process_dev == False:
            for k in range(neg_ratio):
                sample = {
                    'question': queries[i],
                    'passage':negs[k],
                    'label':0
                }
                samples.append(sample)
        else:
            sample = {
                'query': queries[i],
                'positive': [pos_samples[i]],
                'negative': negs 
            }
            samples.append(sample)
        progress_bar2.update(1)
    
    # shuffle samples
    random.shuffle(samples)
    
    with open(cfg.save_path, "w", encoding="utf-8") as json_ds:
        json.dump(samples, json_ds, indent=4, ensure_ascii=False)       

    
    return None


def config_retrieval_pipeline(cfg):
    
    if cfg.retrieval_method == "dpr":
        doc_encoder = cfg.doc_encoder
        query_encoder = cfg.query_encoder

        if os.path.isfile(cfg.index_file):
            # to load the document_store, use below class method
            document_store = FAISSDocumentStore.load(cfg.index_file)

        else:
            raise RuntimeError('FAISS index does not exist, please generate FAISS index first')

        retriever = DensePassageRetriever(document_store=document_store,
                                          top_k = cfg.topk,
                                                  query_embedding_model=query_encoder,
                                                  passage_embedding_model=doc_encoder,
                                                  max_seq_len_query=cfg.max_seq_len_query,
                                                  max_seq_len_passage=cfg.max_seq_len_passage,
                                                  batch_size=cfg.bs,
                                                  use_gpu=False,
                                                  #embed_title=True,
                                                  xlm_roberta=True,
                                                  use_fast_tokenizers=True)
        p = DocumentSearchPipeline(retriever)
        
    elif cfg.retrieval_method == "bm25":
        document_store = ElasticsearchDocumentStore(host=cfg.host, port=cfg.port, index=cfg.index_name)
        retriever = BM25Retriever(document_store)
        p = Pipeline()
        p.add_node(component=retriever, name="Retriever", inputs=["Query"])
        
    else:
        raise NotImplementedError
    return p

def get_hard_negatives(query, p_retrieval, num_retrieve, num_neg, correct_image):
    # hard coded to have BM25 retrieve 200 passages
    res = p_retrieval.run(query=query, params={"Retriever": {"top_k": num_retrieve}})
    #print(res)
    
    text = []
    top_k_doc_links = []
    for doc in res['documents']:
        link = doc.meta['link'].split('/')[-1]
        link = link.split('.')[0]
        top_k_doc_links.append(link)
        text.append(doc.content)
    
    #print(len(text))
    
    if correct_image in top_k_doc_links:
        # there can be multiple instances of correct_image
        text = [t for i, t in enumerate(text) if top_k_doc_links[i] != correct_image]
        
    #print(len(text))  
    assert num_neg <= len(text), 'Did not retrieve enough documents, reduce num_neg'
    return text[:num_neg]


    
    
def extract_text_from_image(image_path, cfg, tokenizer=None, model=None):
    image = preprocess_single_image(cfg, image_path)
    ocr_outputs = apply_tesseract(image, lang = cfg.ocr_lang)
    text = ocr_outputs['text']
    if len(text)>cfg.min_chars:
        docs = postprocess_ocr_outputs_of_single_image(cfg, ocr_outputs, image_path, tokenizer=tokenizer, model=model)

    return docs
            


def parse_cmd():
    desc = 'process DuReader-vis dataset for fine tuning...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    
    args.add_argument('--method', type=str, default='v0', dest='method', help='v0 or v1, v0-use docvqa_train.json text, v1-use our own ocr pipeline')
    args.add_argument('--encoder', type=str, default='dpr', dest='encoder', help='process dataset for which kind of encoder training, dpr or ce')
    args.add_argument('--process_dev', dest='process_dev', help='process dev data for eval during training', action='store_true', default=False)
    
     # directories
    args.add_argument('--data_dir', type=str, default='/home/user/dataset/', dest='data_dir')
    args.add_argument('--train_file', type=str, default='docvqa_train.json', dest='train_file')
    args.add_argument('--dev_file', type=str, default='docvqa_dev.json', dest='dev_file')
    args.add_argument('--save_to', type=str, default='/home/user/output/processed_data/', dest='save_to')
    
    
    # args specifying splitting docs into passages
    args.add_argument('--max_seq_len_passage', type=int, default=500, dest='max_seq_len_passage', help='max seq len for passages')
    args.add_argument('--overlap', type=int, default=10, dest='overlap', help='overlap of text when splitting passages')
    args.add_argument('--min_chars', type=int, default=5, dest='min_chars', help='minimum num of chars a text should have to be counted as a passage')
    
    
    # args for hard negative sampling
    args.add_argument('--retrieval_method', type=str, default=None, dest='retrieval_method', help='dpr, or bm25')
    args.add_argument('--neg_ratio', type=int, default=128, dest='neg_ratio', help='neg:pos ratio')
    args.add_argument('--num_retrieve', type=int, default=200, dest='num_retrieve', help='num of docs to retrieve for sampling hard negatives')
    args.add_argument('--hard_neg', dest='hard_neg', help='sample hard negatives', action='store_true', default=False)
    args.add_argument('--host', type=str, default='localhost', dest='host', help='host ip address of elasticsearch container')
    args.add_argument('--port', type=int, default=9205, dest='port', help='port number of elasticsearch container')
    

    # if using v1 method, args below specify params for the text extraction pipeline
    args.add_argument('--folder_prefix', type=str, default='/home/user/dataset/dureader_vis_images_part_', dest='folder_prefix', help='this arg works with v1 method')
    args.add_argument('--ocr_lang', type=str, default='chi_sim', dest='ocr_lang', help='ocr language')
    args.add_argument('--split_doc', dest='split_doc', help='whether to split doc into shorter passages', action='store_true', default=False)
    args.add_argument('--cluster_doc', dest='cluster_doc', help='whether to cluster lines into passages', action='store_true', default=False)
    args.add_argument('--crop_image', dest='crop_image', help='whether to crop image', action='store_true', default=False)
    args.add_argument('--n_component', type=int, default=2, dest='n_component', help='num of PCA components in clustering line embeddings')
    args.add_argument('--cluster_model', type=str, default='microsoft/infoxlm-base', dest='cluster_model', help='saved dir or pretrained model name on hf')
    args.add_argument('--index_name', type=str, default='faiss', dest='index_name', help='name of table in database, example: faiss')

    
    
    print(args.parse_args())
    return args.parse_args()
    

if __name__ == "__main__":
    
    cfg = parse_cmd()
    
    folder_nums = range(1,11) # corresponds to part 1,2, etc.in dureader-vis
    
    if cfg.encoder == 'dpr':
        print("process training file: {}".format(cfg.train_file))
        data_path=cfg.data_dir+cfg.train_file
        save_path=cfg.save_to+'processed_train.json'
        process_dataset_for_dpr(cfg, data_path, save_path, folder_nums=folder_nums)
        
        if cfg.process_dev==True:
            print("process dev file: {}".format(cfg.dev_file))
            data_path=cfg.data_dir+cfg.dev_file
            save_path=cfg.save_to+'processed_dev.json'
            process_dataset_for_dpr(cfg, data_path, save_path, folder_nums=folder_nums)
            
            
        
    elif cfg.encoder == 'ce':
        process_dataset_for_cross_encoder(cfg)
    else:
        raise NotImplementedError

        
        
        
        
    


    
