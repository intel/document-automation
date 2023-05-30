# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes.retriever.dense import DensePassageRetriever

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, SentenceTransformersRanker
from haystack import Pipeline
from haystack.nodes import JoinDocuments

from utils import MaxSimRanker

import os
import argparse
from tqdm.auto import tqdm
import json

import pandas as pd
import numpy as np

def get_query_dataset(cfg, document_store):
    df = pd.read_csv(cfg.datapath)
    #print('# of rows in df after read: ', df.shape[0])
       
    # since we are testing on subset of data
    # we will only look at in-scope queries, where the correct answer image has been indexed
    if cfg.eval_subset==True:
        existing_docs = get_existing_docs(document_store)
        df = df.loc[df['image_id'].isin(existing_docs)]
        #df = df.sample(cfg.num_query, random_state=100)
    print('# of questions to be tested: ', df.shape[0])
    return df


def run_one_pipeline_to_get_score(query, p1, top_k):
    res1 = p1.run(query=query, params={"Retriever": {"top_k": top_k}})
    scores1 = [p.score for p in res1['documents']]
    images1 = [p.meta['link'] for p in res1['documents']]
    links = []
    for link in images1:
        link = link.split('/')[-1]
        link = link.split('.')[0]
        links.append(link)
        
    assert len(scores1) == len(links), '# of scores not equal to # of images'
    return res1, scores1, links


def get_max_score(img, images, scores):
    if img in images:
        s = [scores[i] for i, x in enumerate(images) if x ==img] # there can be multiple passages with the same img link
        #print('Image: ', img)
        #print('max score of {} scores: {}'.format(len(s), max(s)))
        return max(s)
    else:
        #print('Image: ', img)
        #print('Not in this set')
        return 0.0

def run_hybrid_retrieval_for_one_query(query, p1, p2, weight, top_k, rerank_topk):
    res1, scores1, images1 = run_one_pipeline_to_get_score(query, p1, top_k)
    res2, scores2, images2 = run_one_pipeline_to_get_score(query, p2, top_k)
    
    images = list(set(images1).union(set(images2)))
    #print('total # of images retrieved for this query: ', len(images))
    score1 = []
    score2 = []
    for img in images:
        score1.append(get_max_score(img, images1, scores1))
        score2.append(get_max_score(img, images2, scores2))
        
    #print('bm25 score: ', score1)
    #print('dpr score: ', score2)
    scores = np.array(score1) + weight*np.array(score2)
    #print('final scores: ', scores)
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:rerank_topk]
    #print('indices for top {}:'.format(rerank_topk))
    #print(idx)
    reranked_images = [images[i] for i in idx]
    return reranked_images


def process_single_query(row, p_retrieval, cfg, ranker = None):
    query = row['question']

    if cfg.retrieval_method == 'bm25' or cfg.retrieval_method =='dpr':
        res = p_retrieval.run(query=query, params={"Retriever": {"top_k": cfg.topk}})

        if ranker != None: # use MaxSim reranker implemented by MH
            passages = ranker.get_document_content(res)
            scores = ranker.forward(query, passages) # numpy array, unsorted

            sorted_scores = np.sort(scores)[::-1].tolist() # in descending order, larger score has smaller rank

        top_k_doc_links = []
        for doc in res['documents']:
            link = doc.meta['link'].split('/')[-1]
            link = link.split('.')[0]
            top_k_doc_links.append(link)
 
    elif cfg.retrieval_method == 'hybrid':
        p1, p2 = p_retrieval[0], p_retrieval[1]
        top_k_doc_links = run_hybrid_retrieval_for_one_query(query, p1, p2, cfg.weight, cfg.topk, cfg.rerank_topk)
    
    elif cfg.retrieval_method == "ensemble":
        res = p_retrieval.run(query=query, params={"BM25_Retriever": {"top_k": cfg.topk}, "DPR_Retriever": {"top_k": cfg.topk}})
        #print(len(res['documents']))
        top_k_doc_links = []
        for doc in res['documents']:
            link = doc.meta['link'].split('/')[-1]
            link = link.split('.')[0]
            top_k_doc_links.append(link)
    
    correct_link =row['image_id']

    hit = 0
    out_of_scope = 0
    rank = 20000
    error_sample = None
        
    if correct_link in top_k_doc_links:
        hit = 1
        if ranker != None:
            idx = top_k_doc_links.index(correct_link)
            score = scores[idx]
            rank = 1+sorted_scores.index(score)
        else:    
            rank = 1+top_k_doc_links.index(correct_link)
    else:
        if cfg.error_analysis == True:
            error_sample = {
                'query': query,
                'correct_image':correct_link,
                'top_retrieved_text':[p.content for p in res['documents']],
                'top_retrieved_link':[p.meta['link'] for p in res['documents']],
                'top_retrieved_score':[p.score for p in res['documents']],
                'answer': row['answer']
            }
            
    return hit, out_of_scope, rank, error_sample

def get_existing_docs(document_store):
    existing_docs = []
    try:
        doc_generator = document_store.get_all_documents_generator()
            
        for doc in doc_generator:
            link = doc.meta['link']
            existing_docs.append(link.split('/')[-1].split('.')[0])
        existing_docs = list(set(existing_docs))
        del doc_generator
    except:
        print('Did not get docs from docstore...')
         
    print('There are {} doc images in db...'.format(len(existing_docs)))
    return existing_docs



def compute_recall_at_topk(num_hit, num_query):
    return num_hit/num_query

def comput_mrr(rank_list):
    rank_array = np.array(rank_list)
    sum_reciprocal_rank = np.sum(1.0/rank_array)
    return sum_reciprocal_rank/len(rank_list)


def config_retrieval_pipeline(cfg):
    
    def config_dpr_retrieval(cfg):
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
        
        return p, document_store
    
    def config_bm25_retrieval(cfg):
        document_store = ElasticsearchDocumentStore(host=cfg.host, port=cfg.port, index = cfg.index_name)
        retriever = BM25Retriever(document_store)
        retriever.debug = True
        
        p = Pipeline()
        p.add_node(component=retriever, name="Retriever", inputs=["Query"])
        
        if cfg.ranker_path != None:
            ranker = SentenceTransformersRanker(model_name_or_path=cfg.ranker_path, top_k=cfg.rerank_topk)
            p.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
            
        return p, document_store
        
    def config_join_retrieval(cfg):
        es_document_store = ElasticsearchDocumentStore(host=cfg.host, port=cfg.port, index = cfg.index_name)
        bm25_retriever = BM25Retriever(es_document_store)
        bm25_retriever.debug = True
        
        
        faiss_document_store = FAISSDocumentStore.load(cfg.index_file)
        
        dpr_retriever = DensePassageRetriever(document_store=faiss_document_store,
                                              top_k = cfg.topk,
                                                  query_embedding_model=cfg.query_encoder,
                                                  passage_embedding_model=cfg.doc_encoder,
                                                  max_seq_len_query=cfg.max_seq_len_query,
                                                  max_seq_len_passage=cfg.max_seq_len_passage,
                                                  batch_size=cfg.bs,
                                                  use_gpu=False,
                                                  #embed_title=True,
                                                  xlm_roberta=True,
                                                  use_fast_tokenizers=True)
        
        
        join = JoinDocuments(join_mode="merge", weights=[1, cfg.weight], top_k_join=cfg.rerank_topk)
        
        p = Pipeline()
        p.add_node(component=bm25_retriever, name="BM25_Retriever", inputs=["Query"])
        p.add_node(component=dpr_retriever, name="DPR_Retriever", inputs=["Query"])
        p.add_node(
            component=join, name="JoinResults", inputs=["BM25_Retriever", "DPR_Retriever"]
        )
        return p, es_document_store
    
    
    if cfg.retrieval_method == "dpr":
        p, document_store = config_dpr_retrieval(cfg)             
    elif cfg.retrieval_method == "bm25":
        p, document_store = config_bm25_retrieval(cfg)
        
    elif cfg.retrieval_method == "hybrid":
        # get two retrieval pipelines
        p1, document_store = config_bm25_retrieval(cfg)
        p2, _ = config_dpr_retrieval(cfg)
        p = [p1, p2]
    elif cfg.retrieval_method == "ensemble":
        p, document_store = config_join_retrieval(cfg)
        
    else:
        raise NotImplementedError
    
    
    
    return p, document_store
    


def simple_retrieval_test(query, top_k):
    p_retrieval, document_store = config_retrieval_pipeline(cfg)
    res = p_retrieval.run(query=query, params={"Retriever": {"top_k": top_k}})
    print_documents(res, max_text_len=200)
    


def test_retrieval_pipeline(cfg): 
    
    p_retrieval, document_store = config_retrieval_pipeline(cfg)
    
    #MaxSim ranker implemented by MH
    ranker = None
    
    df = get_query_dataset(cfg, document_store)

    sum_hit = 0
    sum_out_of_scope = 0
    rank_list = []
    
    num_query = df.shape[0]
    
    if cfg.error_analysis == True:
        error_samples = []
        error_query = []
    
    progress_bar=tqdm(range(num_query))
    
    for _, row in df.iterrows():
        hit, out_of_scope, rank, sample = process_single_query(row, p_retrieval, cfg, ranker)
        sum_hit+=hit
        sum_out_of_scope += out_of_scope
        rank_list.append(rank)
        if sample != None:
            error_samples.append(sample)
            error_query.append(sample['query'])
        progress_bar.update(1)
    
    recall = compute_recall_at_topk(sum_hit, num_query)
    mrr = comput_mrr(rank_list)
    
    

    if cfg.retrieval_method=="ensemble" or cfg.retrieval_method=="hybrid":
        print('hit at top {}: {}'.format(cfg.rerank_topk, sum_hit))
        print('Recall at top {}: {:.4f}'.format(cfg.rerank_topk, recall))
        print('MRR at top {}: {:.4f}'.format(cfg.rerank_topk, mrr))
    else:
        print('hit at top {}: {}'.format(cfg.topk, sum_hit))
        print('Recall at top {}: {:.4f}'.format(cfg.topk, recall))
        print('MRR at top {}: {:.4f}'.format(cfg.topk, mrr))
    
    if cfg.error_analysis == True:
        #with open('error_query.txt', 'w') as f:
        #    f.write(error_query)
        print(error_query)
            
        with open(cfg.save_path, "w", encoding="utf-8") as json_ds:
            json.dump(error_samples, json_ds, indent=4, ensure_ascii=False)
    return recall, mrr
    

def find_best_param(param_list, cfg):
    recall_list = []
    mrr_list = []
    
    for w in param_list:
        cfg.weight=w
        recall, mrr = test_retrieval_pipeline(cfg)
        recall_list.append(recall)
        mrr_list.append(mrr)
    print('recall: ', recall_list)
    print('mrr: ', mrr_list)
        
    
    

def parse_cmd():
    desc = 'evaluate retrieval performance for DuReader-vis dataset...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    
    args.add_argument('--retrieval_method', type=str, default=None, dest='retrieval_method', help='dpr, or bm25, ensemble')
    args.add_argument('--topk', type=int, default=5, dest='topk', help='num of docs to be retrieved for each query')
    args.add_argument('--index_name', type=str, default='faiss', dest='index_name', help='name of table in database, example: faiss')
    args.add_argument('--datapath', type=str, default=None, dest='datapath', help='path to the docvqa_dev.csv file')
    #args.add_argument('--db', type=str, default=None, dest='db', help='database path')
    
    # args for BM25 retrieval
    args.add_argument('--host', type=str, default='localhost', dest='host', help='host ip address of elasticsearch container')
    args.add_argument('--port', type=int, default=9200, dest='port', help='port number of elasticsearch container')
    
    # args for dpr based retrieval
    args.add_argument('--index_file', type=str, default=None, dest='index_file', help='filename of faiss index, example: faiss-index-so.faiss')
    args.add_argument('--query_encoder', type=str, default=None, dest='query_encoder', help='saved dir or pretrained model name on hf')
    args.add_argument('--doc_encoder', type=str, default=None, dest='doc_encoder', help='saved dir or pretrained model name on hf')
    args.add_argument('--max_seq_len_passage', type=int, default=512, dest='max_seq_len_passage', help='max seq len for passage encoder')
    args.add_argument('--max_seq_len_query', type=int, default=128, dest='max_seq_len_query', help='max seq len for query encoder')
    args.add_argument('--bs', type=int, default=16, dest='bs', help='batch size for DPR')
    
    # args for hybrid retrieval
    args.add_argument('--rerank_topk', type=int, default=5, dest='rerank_topk', help='num of docs to be retrieved for each query')
    args.add_argument('--weight', type=float, default=None, dest='weight', help='weight factor for dpr score in hybrid retrieval')
    args.add_argument('--hpo', dest='hpo', help='do hyperparm search', action='store_true', default=False)
    
 
    # args for debugging and error analysis
    args.add_argument('--eval_subset', dest='eval_subset', help='evaluate retrieval perf on subset of data', action='store_true', default=False)
    args.add_argument('--simple_test', dest='simple_test', help='do a simple retrieval test', action='store_true', default=False)
    args.add_argument('--num_query', type=int, default=100, dest='num_query', help='num of queries to be tested')
    args.add_argument('--error_analysis', dest='error_analysis', help='do error_analysis', action='store_true', default=False)
    args.add_argument('--saveto', type=str, default=None, dest='save_path', help='save to path for error analysis results')
    args.add_argument('--ranker_path', type=str, default=None, dest='ranker_path', help='abs path to fine-tuned ranker')
    
    
    
    print(args.parse_args())
    return args.parse_args()
    

if __name__ == "__main__":
    
    cfg = parse_cmd()
    
    if cfg.simple_test:
        query = "how long do cars last"
        simple_retrieval_test(query, cfg.topk)
    elif cfg.hpo:
        param_list = np.linspace(0.5, 1.5, num=3)
        find_best_param(param_list, cfg)
        
    else: 
        test_retrieval_pipeline(cfg)
        
        
        
        
        
        
        
        
        
