# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
from typing import Dict, Iterable, Optional, Union
import pytesseract
from PIL import Image
import glob
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
#import cv2 as cv
from sklearn.cluster import KMeans
#from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


###------------ preprocessing ------------------------------
def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def crop_image(image):
    # image: PIL Image object
    width, height = image.size
    #print('image width = {}, height = {}'.format(width, height))
 
    left = 0
    top = int(height *0.05)
    right = int(width*0.70)
    bottom = int(0.9* height)
 
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = image.crop((left, top, right, bottom))
    return im1

def binarize_image(img):
    '''
    img: np array
    '''
    #img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    #norm_img = np.zeros((img.shape[0], img.shape[1]))
    #img = cv.normalize(img, norm_img, 0, 255, cv.NORM_MINMAX)
    img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,2)
    return Image.fromarray(img)
    

def preprocess_single_image(cfg, image_path):
    # preprocess contains steps to be done before OCR
    # step 1: binarize
    # Step 2: crop - this is a heuristic to remove ads
    # Step 3: TODO - other image transforms
            
    image_file = Image.open(image_path)
    # Grayscale

    image_file = image_file.convert('L')
    
    if cfg.crop_image == True:
        image_file = crop_image(image_file)
    
    return image_file

def gray_scale_image_with_opencv(img_path):
    #img = cv.imread(img_path, cv.IMREAD_COLOR) #cv.IMREAD_GRAYSCALE)
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    #norm_img = np.zeros((img.shape[0], img.shape[1]))
    #img = cv.normalize(img, norm_img, 0, 255, cv.NORM_MINMAX)
    #img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #            cv.THRESH_BINARY,11,2)
    
    #return Image.fromarray(img)
    return img
#----------------- OCR -----------------------------------

def apply_tesseract(pil_image, lang: Optional[str]):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""

    # apply OCR
    #pil_image = to_pil_image(image)
    image_width, image_height = pil_image.size
    data = pytesseract.image_to_data(pil_image, lang=lang, output_type="dict")
    # ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text']
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]
    line_num = data['line_num']
    block_num = data['block_num']

    # filter empty words and corresponding coordinates
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]
    
    line_num = [coord for idx, coord in enumerate(line_num) if idx not in irrelevant_indices]
    block_num = [coord for idx, coord in enumerate(block_num) if idx not in irrelevant_indices]

    # turn coordinates into (left, top, left+width, top+height) format
    actual_boxes = []
    for x, y, w, h in zip(left, top, width, height):
        actual_box = [x, y, x + w, y + h]
        actual_boxes.append(actual_box)

    # finally, normalize the bounding boxes
    normalized_boxes = []
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"
    
    text = ''.join(words)
    
    outputs = {
        'text':text,
        'bbox':normalized_boxes,
        'line_num':line_num,
        'block_num':block_num,
        'words':words
    
    }

    return outputs


def apply_paddleocr(ocr, img, lang='ch'):
    '''
    args:
    ocr - PaddleOCR object
    img - numpy array
    lang - `ch`, `en`, `fr`, `german`, `korean`, `japan`
    '''
    result = ocr.ocr(img, cls=False)
    txts = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            txt = line[1][0].strip()
            #print(txt)
            if contain_ads(txt)==False:
                txts.append(txt)
            
    #boxes = [line[0] for line in result]
    #txts = [line[1][0] for line in result]
    #print(txts)
    #scores = [line[1][1] for line in result]
    #txts = [t.strip() for t in txts]
    text = ''.join(txts)
    outputs = {
        'text':text,
        'bbox':None,
        'line_num':None,
        'block_num':None,
        'words':txts
    
    }
    return outputs

#--------------- post processing --------------------------------
def get_split(text, cfg):
    l_total = []
    l_parcial = []
    non_overlap_len = cfg.max_seq_len_passage-cfg.overlap
    
    if len(text)<=cfg.max_seq_len_passage:
        l_total.append(text)
    else:
        # len(text) > max_len > non_overlap_len
        n = len(text)//non_overlap_len # n>=1 for sure in this case
        for w in range(n):
            if w == 0:
                l_parcial = text[:cfg.max_seq_len_passage]
            else:
                l_parcial = text[w*non_overlap_len:w*non_overlap_len + cfg.max_seq_len_passage]
            l_total.append(l_parcial)
        # remainder text
        remainder = text[cfg.max_seq_len_passage+(n-1)*non_overlap_len:]
        if len(remainder)>cfg.min_chars:
            l_total.append(remainder)

    return l_total


def contain_ads(words):
    ads = ['广告', '热门']
    for ad in ads:
        if ad in words:
            print('contain ad')
            return True
    return False

def get_words_by_block_and_line(words, line_num, block_num):
    assert len(words) == len(line_num), '# of words not equal to # line_num'
    assert len(words) == len(block_num), '# of words not equal to # block_num'
    word_on_this_line =words[0]
    words_by_line = []
    for i in range(1, len(line_num)):
        pre_b = block_num[i-1]
        pre_l = line_num[i-1]
        b = block_num[i]
        l = line_num[i]
        if pre_b == b and pre_l == l:
            word_on_this_line += words[i]
                       
        else:
            # new line
            if contain_ads(word_on_this_line) == False:
                words_by_line.append(word_on_this_line)
            
            word_on_this_line = words[i]
        #print('word_on_this_line:', word_on_this_line)
        #print('words_by_line: ', words_by_line)
            
    #print(words_by_line)
    return words_by_line
            
    
def get_embedding_by_unit(words_by_unit, tokenizer, max_seq_len, model):
    tokenized = tokenizer(words_by_unit, return_tensors="pt", padding="max_length", 
                                 truncation=True, max_length = max_seq_len)
    outputs = model(**tokenized)
    embeddings = outputs.last_hidden_state[:,0,:]
    return embeddings.detach().numpy()


def get_text_clusters(words_by_line, embeddings_lines, n_components, n_cluster):
    if n_cluster > len(words_by_line):
        print('In total {} units of words, no clustering needed'.format(len(words_by_line)))
        return words_by_line
    else:
        pca = PCA(n_components=n_components)
        pca_embeddings_lines = pca.fit(embeddings_lines).transform(embeddings_lines)
        print('shape of pca embeddings: ', pca_embeddings_lines.shape)


        kmeans=KMeans(n_clusters=n_cluster)
        embedding_clusters = kmeans.fit(pca_embeddings_lines)
        #embedding_clusters = DBSCAN().fit(pca_embeddings_lines)
        text_cluster = ['']*n_cluster
        for i, w in enumerate(words_by_line):
            #print(w)
            c = embedding_clusters.labels_[i]
            #print(c)
            text_cluster[c]+=w

        return text_cluster
    
def postprocess_ocr_by_clustering_line_embedding(ocr_outputs, tokenizer, model, n_components, passage_len=None, ocr_engine='tesseract'):
    if ocr_engine == "tesseract":
        words = ocr_outputs['words']
        line_num = ocr_outputs['line_num']
        block_num = ocr_outputs['block_num']
        words_by_line = get_words_by_block_and_line(words, line_num, block_num)
    elif ocr_engine == "paddleocr":
        words_by_line = ocr_outputs['words']
    else:
        raise RuntimeError('ocr_engine must be either tesseract or paddleocr')
        
    max_seq_len = 0
    for w in words_by_line:
        if len(w) > max_seq_len:
            max_seq_len = len(w)
    print('max seq len: ', max_seq_len)
    #pretrained_model_name = 'microsoft/infoxlm-base'

    embeddings = get_embedding_by_unit(words_by_line, tokenizer, max_seq_len, model)
    print('shape of original embeddings: ', embeddings.shape)
    
    if passage_len != None:
        text = ''.join(words_by_line)
        n_cluster = 1+len(text)//passage_len
    #if n_cluster < 2:
    #    n_cluster = 2
    else:
        n_cluster = 2
    print('# of clusters: ', n_cluster)
    
    text_cluster = get_text_clusters(words_by_line, embeddings, n_components, n_cluster)
    return text_cluster


def postprocess_ocr_outputs_of_single_image(cfg, ocr_outputs, path, tokenizer= None, model=None):
    '''
    ocr_outputs: dict
     outputs = {
        'text':text, ocr text pieced together
        'bbox':None,
        'line_num':None,
        'block_num':None,
        'words':direct outputs from ocr, words by identified blocks
    
    }
    '''    
            
    if cfg.force_num_cluster==True:
        passage_len = None
    else:
        passage_len = cfg.max_seq_len_passage
        
    if cfg.split_doc == True and cfg.cluster_doc == True:
        text_cluster = postprocess_ocr_by_clustering_line_embedding(ocr_outputs, tokenizer, model, 
                                                                    cfg.n_components, passage_len)
        text_splits = []
        for t in text_cluster:
            if len(t) > cfg.min_chars:
                text_splits += get_split(t, cfg)
    
        docs = [{'content':text_splits[i], 
            'meta':{'link': path.split('/')[-1]}} for i in range(len(text_splits))]
 
    elif cfg.split_doc == True and cfg.cluster_doc == False:
        # simple splitting with overlap 
        text = ocr_outputs['text']
        text_batch = []
        num_passages = 0
        if len(text) > cfg.min_chars:
            passages = get_split(text, cfg)
                    #print('passages:\n',passages)
            num_passages += len(passages)
            text_batch += passages

        assert len(text_batch) == num_passages, 'length of text_batch not equal to num_passages'

        docs = [{'content':text_batch[i], 
            'meta':{'link': path.split('/')[-1]}} for i in range(num_passages)]
        
    elif cfg.split_doc == False and cfg.cluster_doc == True:
        # cluster words by line based on LM-generated embeddings
        text_cluster = postprocess_ocr_by_clustering_line_embedding(ocr_outputs, tokenizer, model, 
                                                                    cfg.n_components, passage_len)
        docs = [{'content':text_cluster[i], 
            'meta':{'link': path.split('/')[-1]}} for i in range(len(text_cluster))]
    
    else:
        # no post processing
        docs = [{'content':ocr_outputs['text'], 'meta':{'link': path.split('/')[-1]}}]
    print('Image {} is split into {} passages'.format(path.split('/')[-1],len(docs)))
    return docs    
    
#--------------------------------------------------------------------------------------------

def get_existing_docs(document_store):
    existing_docs = []
    try:
        doc_generator = document_store.get_all_documents_generator()
            
        for doc in doc_generator:
            existing_docs.append(doc.meta['link'])
        existing_docs = list(set(existing_docs))
        del doc_generator
    except:
        print('Did not get docs from docstore...')
            
            
    #print('There are {} doc images in db...'.format(len(existing_docs)))
    return existing_docs


def get_all_image_path(folder_nums, folder_prefix):
    full_path_all = []
    img_id_all = []
    for i in folder_nums:
        folder = folder_prefix + str(i) + '/'
        full_path_all += glob.glob(folder+'*.png')
    print('total # images: ', len(full_path_all))
    print('full path example: ', full_path_all[0])
    for p in full_path_all:
        img_id_all.append(p.split('/')[-1].split('.')[0])
    print('image id example: ', img_id_all[0])
    #idx = img_id_all.index(image_id)
    return full_path_all, img_id_all


class MaxSimRanker():
    def __init__(self, cfg):
        self.cfg = cfg
        self.doc_tokenizer = AutoTokenizer.from_pretrained(self.cfg.doc_encoder)
        self.query_tokenizer = AutoTokenizer.from_pretrained(self.cfg.query_encoder)
        self.doc_encoder = AutoModel.from_pretrained(self.cfg.doc_encoder)
        self.query_encoder = AutoModel.from_pretrained(self.cfg.query_encoder)
        
    # adapt from https://huggingface.co/sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco
    def forward(self,
                query,
                document):
        
        # query: string
        # document: list of strings
        
        query = [query]*len(document) # query and documents need to of same batch size
        
        query_tokens = self.query_tokenizer(query, padding="max_length", return_tensors="pt",
                                 truncation=True, max_length = self.cfg.max_seq_len_query)
        
        doc_tokens = self.doc_tokenizer(document, return_tensors="pt", padding="max_length", 
                                 truncation=True, max_length = self.cfg.max_seq_len_passage)

        query_vecs = self.forward_representation(query_tokens, sequence_type = "query_encode")
        document_vecs = self.forward_representation(doc_tokens, sequence_type = "doc_encode")

        score = self.forward_aggregation(query_vecs,document_vecs,query_tokens["attention_mask"],doc_tokens["attention_mask"])
        return score
    
    def get_document_content(self, res):
        # red: results returned by haystack retriever
        #print(res['documents'])
        text = [doc.content for doc in res['documents']]
        return text
        

    def forward_representation(self,
                               tokens,
                               sequence_type=None) -> torch.Tensor:
        
        if sequence_type == "doc_encode":
            model = self.doc_encoder
        elif sequence_type == "query_encode":
            model = self.query_encoder
        
        model.eval()
        vecs = model(**tokens)[0] # get the hidden state from last layer
        #vecs = self.compressor(vecs)

        # if encoding only, zero-out the mask values so we can compress storage
        #if sequence_type == "doc_encode" or sequence_type == "query_encode": 
        #    vecs = vecs * tokens["tokens"]["mask"].unsqueeze(-1)

        return vecs

    def forward_aggregation(self,query_vecs, document_vecs,query_mask,document_mask):
        
        # create initial term-x-term scores (dot-product)
        score = torch.bmm(query_vecs, document_vecs.transpose(2,1))

        # mask out padding on the doc dimension (mask by -1000, because max should not select those, setting it to 0 might select them)
        exp_mask = document_mask.bool().unsqueeze(1).expand(-1,score.shape[1],-1)
        score[~exp_mask] = - 10000

        # max pooling over document dimension
        score = score.max(-1).values

        # mask out paddding query values
        score[~(query_mask.bool())] = 0

        # sum over query values
        score = score.sum(-1)
        #print('score before numpy: ', score)
        # convert to numpy and then to list
        score = score.detach().cpu().numpy().tolist()
        #print('score after numpy: ', score)

        return score