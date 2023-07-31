# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
import ray
import os
import base64
from typing import Dict, List
import numpy as np
from ray.data import ActorPoolStrategy
from ray.data.dataset_pipeline import DatasetPipeline
from haystack.document_stores import FAISSDocumentStore, ElasticsearchDocumentStore
from haystack.nodes.retriever.dense import DensePassageRetriever
import argparse, time, os
import pytesseract
from PIL import Image
from io import BytesIO
from utils import crop_image, binarize_image, apply_tesseract, apply_paddleocr
from utils import postprocess_ocr_outputs_of_single_image
import pandas as pd
import faiss
from transformers import AutoModel, AutoTokenizer
from paddleocr import PaddleOCR

class EmbeddingTask :
    def __init__(self, config):
        self.cfg = config
        doc_encoder = self.cfg.doc_encoder #"microsoft/infoxlm-base"
        query_encoder = self.cfg.query_encoder #"microsoft/infoxlm-base"
        
            
            
        self.retriever = DensePassageRetriever(document_store=None,
                                              query_embedding_model=query_encoder,
                                              passage_embedding_model=doc_encoder,
                                              max_seq_len_query=self.cfg.max_seq_len_query,
                                              max_seq_len_passage=self.cfg.max_seq_len_passage,
                                              batch_size=self.cfg.embedding_bs,
                                              use_gpu=False,
                                              xlm_roberta=True,
                                              use_fast_tokenizers=True)
        
 

    def __call__(self, docs) :
        embeddings = self.retriever.embed_documents(docs)
        assert len(embeddings) == len(docs), 'length of embeddings not equal to docs'
        result = [{'doc_id': docs[index].id, 'embedding': embedding} for index, embedding in enumerate(embeddings)]
        return result

class PreprocessTask :
    def __init__(self, config):
        import pytesseract
        from PIL import Image
        
        self.cfg = config
        if self.cfg.cluster_doc == True:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.cluster_model)
            self.model = AutoModel.from_pretrained(self.cfg.cluster_model)
        else:
            self.tokenizer = None
            self.model = None
            
        if self.cfg.ocr_engine=="paddleocr":
            self.pocr = PaddleOCR(use_angle_cls=True, lang='ch', 
                det_limit_type='min',
                det_limit_side_len=2048,
                show_log =False,
                #cpu_threads=20, #would this cause issues when Ray assigns fewer threads?
               ) # need to run only once to download and load model into memory


    def __call__(self, data) :
        path, image = data[0]
        print(f"path = {path.split('/')[-1]}")
        docs = []
        try :
            # image preprocessing
            if self.cfg.preprocess == 'grayscale':
                image = Image.open(BytesIO(image)).convert('L')
            elif self.cfg.preprocess == 'binarize':
                image = np.asarray(Image.open(BytesIO(image)).convert('L'))
                image = binarize_image(image)
            else:
                image = Image.open(BytesIO(image)).convert('RGB')
                
            if self.cfg.crop_image == True:
                image = crop_image(image)
                
            # OCR
            if self.cfg.ocr_engine == "tesseract":
                ocr_outputs = apply_tesseract(image, lang = self.cfg.ocr_lang)    
            elif self.cfg.ocr_engine == "paddleocr":
                image = np.asarray(image) # convert PIL to nparray
                ocr_outputs = apply_paddleocr(self.pocr, image)     
            else:
                raise NotImplementedError
                
            # post processing ocr outputs
            text = ocr_outputs['text']
            if len(text) > self.cfg.min_chars:
                docs = postprocess_ocr_outputs_of_single_image(self.cfg, ocr_outputs, path, tokenizer= self.tokenizer, model=self.model)
            
        except Exception as err:
            print(f"exception: {err=}, {type(err)=}")
            print(f"broken_image= {path}")
        finally: 
            return docs

        
def connect_to_esdocstore(cfg):
    return ElasticsearchDocumentStore(
                    host=cfg.esdb, username="", password="",
                    index=cfg.index_name)

def connect_to_faissdocstore(cfg):
    if os.path.isfile(cfg.index_file):
        dpr_document_store = FAISSDocumentStore.load(cfg.index_file)

    else:
        dpr_document_store = FAISSDocumentStore(
                    sql_url = cfg.db,#'postgresql://postgres:postgres@10.165.9.52:5432/haystack', 
                    faiss_index_factory_str="HNSW",
                    return_embedding=False, 
                    progress_bar=True,
                    index=cfg.index_name, 
                    n_links = cfg.faiss_nlinks, #64, #512,
                    ef_search= cfg.faiss_efsearch, #20, #128,
                    ef_construction= cfg.faiss_efconstruct, #80, #200,
                    validate_index_sync=False)
        
    return dpr_document_store
    
        
class IndexingTask :
    def __init__(self, config):
        self.cfg = config
        if config.retrieval_method == "dpr":
            self.dpr_document_store = connect_to_faissdocstore(self.cfg)
                
        elif config.retrieval_method == "bm25":
            self.bm25_document_store = connect_to_esdocstore(self.cfg)
        elif config.retrieval_method == "all":
            self.bm25_document_store = connect_to_esdocstore(self.cfg)
            self.dpr_document_store = connect_to_faissdocstore(self.cfg)
    
    
    def _write_docs(self, docs: List[Dict]) :
        if self.cfg.retrieval_method == "dpr":
            print("write to postgresql")
            self.dpr_document_store.write_documents(docs, index=self.cfg.index_name)
        elif self.cfg.retrieval_method == "bm25":
            print("write to es")
            self.bm25_document_store.write_documents(docs, index=self.cfg.index_name)
        elif self.cfg.retrieval_method == "all":
            print("write to es and postgresql")
            self.dpr_document_store.write_documents(docs, index=self.cfg.index_name)
            self.bm25_document_store.write_documents(docs, index=self.cfg.index_name)
    
    def __call__(self, docs : List[Dict]):
        try :
            self._write_docs(docs)
        except Exception as err:
            print(f"exception: {err=}, {type(err)=}, try again")
            self._write_docs(docs)
        finally:   
            return [len(docs)]



        
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GenDocStoreBase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.document_store = None
    
       
        
    def _update_embeddings(self, docs, vector_id) :
        vector_id_map = {}
        embeddings =  []
        for doc in docs:
            vector_id_map[str(doc['doc_id'])] = str(vector_id)
            embeddings.append(doc['embedding'])
            vector_id += 1
        embeddings_to_index = np.array(embeddings, dtype="float32")
        self.document_store.faiss_indexes[self.cfg.index_name].add(embeddings_to_index)
        self.document_store.update_vector_ids(vector_id_map, index=self.cfg.index_name)
        return vector_id

    
    def gen_docstore(self):
        DATASET_ROOT_PATH = "/home/user/dataset/"
        dataset_dirs = os.listdir(DATASET_ROOT_PATH)
        dataset_dirs = [ DATASET_ROOT_PATH + x   for x in dataset_dirs if os.path.isdir( DATASET_ROOT_PATH + x ) ]
        if len(dataset_dirs) == 0 :
            dataset_dirs = [DATASET_ROOT_PATH]
            
            
        if self.cfg.add_doc:
            time0 = time.time()
            for dataset_dir in dataset_dirs :
                print(f'dir_path={dataset_dir}')
                ds = ray.data \
                    .read_binary_files(paths = dataset_dir, include_paths = True)
                
                if self.cfg.toy_example==True:
                    ds=ds.limit(10)
                print(ds)
                time1 = time.time()
                
                ds = ds.map_batches(PreprocessTask, 
                                    compute=ActorPoolStrategy(self.cfg.preprocess_min_actors, self.cfg.preprocess_max_actors), 
                                    batch_size=1, 
                                    num_cpus=self.cfg.preprocess_cpus_per_actor, 
                                    fn_constructor_args = [self.cfg]) # 1 node. no clustering: (16, 80) actors, num_cpu = 1
                
                time2 = time.time()
                print(ds)
                
                
                ds = ds.map_batches(IndexingTask, 
                                    compute=ActorPoolStrategy(1, 1), 
                                    batch_size=self.cfg.writing_bs, 
                                    num_cpus=self.cfg.writing_cpus_per_actor, 
                                    fn_constructor_args = [self.cfg]) # 1 node
                #ds.show()
                time3 = time.time()
                print(f'preprocess time= {time2-time1}')
                print(f'write doc time= {time3-time2}')
                
        if self.cfg.embed_doc:

            if self.cfg.retrieval_method == "dpr" or self.cfg.retrieval_method == "all":
                #from haystack.document_stores import FAISSDocumentStore
                self.document_store = connect_to_faissdocstore(self.cfg)
                
                time4 = time.time()
                docs = self.document_store.get_all_documents(index = self.cfg.index_name, return_embedding = False)
                print(len(docs))
                ds = ray.data.from_items(docs)
                print(ds)
                
                ds = ds.map_batches(EmbeddingTask, 
                                    batch_size = self.cfg.embedding_bs, 
                                    compute=ActorPoolStrategy(self.cfg.embedding_min_actors, self.cfg.embedding_max_actors), 
                                    num_cpus=self.cfg.embedding_cpus_per_actor, 
                                    fn_constructor_args = [self.cfg]) # 1 node original batch_size = 50, (4, 8), num_cpus=5
                time5 = time.time()
                vector_id = 0
                for batch in ds.iter_batches():
                    print(len(batch)) 
                    vector_id = self._update_embeddings(docs = batch, vector_id = vector_id)
                self.document_store.save(self.cfg.index_file)
                time6 = time.time()
                print(f'embedding time= {time5-time4}')
                print(f'save time= {time6-time5}')
            else:
                raise RuntimeError('Retrieval method must be DPR or all in order to embed documents into faiss indices')
            #time7 = time.time()
            #print(f'total time= {time7-time0}')
        
        


        

        
def parse_cmd():
    desc = 'generate documentstore for DuReader-vis dataset...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    
    # args for controlling indexing flow
    args.add_argument('--retrieval_method', type=str, default='all', dest='retrieval_method', help='dpr, or bm25, or all')
    args.add_argument('--add_doc', dest='add_doc', help='add docs to database', action='store_true', default=False)
    args.add_argument('--embed_doc', dest='embed_doc', help='embed docs in database', action='store_true', default=False)
    #args.add_argument('--update_existing', dest='update_existing', help='whether to update exisiting embeddings', action='store_true', default=False)
    args.add_argument('--toy_example', dest='toy_example', help='run a toy example with a small subset of images', action='store_true', default=False)
    

    # image preprocessing param
    args.add_argument('--preprocess', type=str, default='grayscale', dest='preprocess', help='image preprocessing method: grayscale, binarize, or none')
    args.add_argument('--crop_image', dest='crop_image', help='whether to crop image', action='store_true', default=False)
    
    # ocr params
    args.add_argument('--ocr_lang', type=str, default='chi_sim', dest='ocr_lang', help='ocr language for tesseract ocr')
    args.add_argument('--ocr_cfg', type=str, default=None, dest='ocr_cfg', help='custom ocr config for tesseract ocr')
    args.add_argument('--ocr_engine', type=str, default='paddleocr', dest='ocr_engine', help='tesseract or paddleocr')
    
    # post processing params
    args.add_argument('--max_seq_len_passage', type=int, default=500, dest='max_seq_len_passage', help='max seq len for passage encoder')
    args.add_argument('--max_seq_len_query', type=int, default=128, dest='max_seq_len_query', help='max seq len for query encoder')
    args.add_argument('--overlap', type=int, default=10, dest='overlap', help='overlap of text when splitting passages')
    #args.add_argument('--num_docs', type=int, default=4, dest='num_docs', help='num of docs to be indexed')
    args.add_argument('--min_chars', type=int, default=5, dest='min_chars', help='minimum num of chars a doc should have to be included in docstore')
    args.add_argument('--split_doc', dest='split_doc', help='whether to split doc into shorter passages', action='store_true', default=False)
    args.add_argument('--cluster_doc', dest='cluster_doc', help='whether to cluster lines into passages', action='store_true', default=False)
    args.add_argument('--n_components', type=int, default=2, dest='n_components', help='num of PCA components in clustering line embeddings')
    args.add_argument('--cluster_model', type=str, default='microsoft/infoxlm-base', dest='cluster_model', help='saved dir or pretrained model name on hf')
    args.add_argument('--force_num_cluster', dest='force_num_cluster', help='whether to force num_cluster=2', action='store_true', default=False)

    
    # database and indexing params
    args.add_argument('--db', type=str, default=None, dest='db', help='database path')
    args.add_argument('--esdb', type=str, default=None, dest='esdb', help='elasticsearch database path') 
    args.add_argument('--index_name', type=str, default='document', dest='index_name', help='name of table in database, example: faiss')
    # the following args are specific to FaissDocumentStore
    args.add_argument('--index_file', type=str, default='/home/user/output/index_files/faiss-index-so.faiss', dest='index_file', help='filename of faiss index, example: faiss-index-so.faiss')
    args.add_argument('--faiss_nlinks', type=int, default=512, dest='faiss_nlinks', help='n_links param for faiss document store')
    args.add_argument('--faiss_efsearch', type=int, default=128, dest='faiss_efsearch', help='ef_search param for faiss document store')
    args.add_argument('--faiss_efconstruct', type=int, default=200, dest='faiss_efconstruct', help='ef_construction param for faiss document store')

    
    # DPR models to generate faiss indices
    args.add_argument('--query_encoder', type=str, default='/home/user/output/dpr_models/query_encoder', dest='query_encoder', help='saved dir or pretrained model name on hf')
    args.add_argument('--doc_encoder', type=str, default='/home/user/output/dpr_models/passage_encoder', dest='doc_encoder', help='saved dir or pretrained model name on hf')
    #args.add_argument('--bs', type=int, default=50, dest='bs', help='DPR batch size')
    
    # Ray params
    args.add_argument('--writing_bs', type=int, default=10000, dest='writing_bs', help='batch size for writing passages into database')
    args.add_argument('--embedding_bs', type=int, default=50, dest='embedding_bs', help='batch size for embedding passages with DPR')
    args.add_argument('--preprocess_min_actors', type=int, default=8, dest='preprocess_min_actors', help='min ray actors for preprocessing task')
    args.add_argument('--preprocess_max_actors', type=int, default=20, dest='preprocess_max_actors', help='max ray actors for preprocessing task')
    args.add_argument('--embedding_min_actors', type=int, default=4, dest='embedding_min_actors', help='min ray actors for embedding task')
    args.add_argument('--embedding_max_actors', type=int, default=8, dest='embedding_max_actors', help='max ray actors for embedding task')
    
    args.add_argument('--preprocess_cpus_per_actor', type=int, default=4, dest='preprocess_cpus_per_actor', help='num of cpus per ray actor for preprocessing task')
    args.add_argument('--writing_cpus_per_actor', type=int, default=4, dest='writing_cpus_per_actor', help='num of cpus per ray actor for writing passages into database')
    args.add_argument('--embedding_cpus_per_actor', type=int, default=10, dest='embedding_cpus_per_actor', help='num of cpus per ray actor for embedding task')
    
    
    
    
    
 
    
    print(args.parse_args())
    return args.parse_args()


if __name__ == "__main__":
    config = parse_cmd()
    ray.init(address='auto')
    index_generator = GenDocStoreBase(config)
    index_generator.gen_docstore()
        
 