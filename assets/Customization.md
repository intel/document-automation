# How to customize Document Automation use case

## Customize DPR fine tuning pipeline
The following hyperparameters can be customized through command line arguments in the ```scripts/run_dpr_training.sh``` script.

| Command line argument | What it is for | Default value | Data type |
|-----------------------|--------------|-------------|---------|
| --num_hard_neg | number of hard negative samples per positive sample in DPR fine tuning | 0 | int |
| --bs | batch size in DPR fine tuning | 128 | int |
| --epochs|number of epochs|3|int|
| --eval_every|evaluate DPR performance every specified number of training steps|87|int|
| --lr|learning rate|learning rate of AdamW optimizer|1e-5|float|
| --warmup|number of warmup steps|20|int|
| --query_encoder|pretrained model name on HF model hub as the starting point of query encoder|microsoft/infoxlm-base|any model on Huggingface model hub that can be loaded by the AutoModel.from_pretrained method|
| --doc_encoder|pretrained model name on HF model hub as the starting point of document encoder|microsoft/infoxlm-base|any model on Huggingface model hub that can be loaded by the AutoModel.from_pretrained method|
| --max_len_query|maximum sequence length of the query|64|int|
| --max_len_passage|maximum sequence length of the document passage|500|int|

There are some other hyperparameters that can be further customized through the haystack DPR APIs. Please refer to the [haystack source code](https://github.com/deepset-ai/haystack/blob/main/haystack/nodes/retriever/dense.py) for more information.


## Customize indexing pipeline
You can use command line arguments in the ```scripts/run_distributed_indexing.sh``` to customize the indexing pipeline with the functionalities already implemented in our use case. Advanced users can write their own methods and incorporate customized functionality into src/gen-sods-doc-image-ray.py.
### Command line arguments for controlling the indexing pipeline
| Command line argument | Options | What it does|
|-----------------------|---------|-------------|
|--retrieval_method| all, dpr, bm25| ```all```: index into both Elasticsearch and postgresql databases for BM25 retrieval and DPR retrieval. ```dpr```: only index into postgresql database. ```bm25```: only index into Elasticsearch database.|
|--add_doc|this is a "store_true" flag, you either include it or not in the command|if included, the pipeline will add documents into database(s)|
|--embed_doc|this is a "store_true" flag, you either include it or not in the command|if included, the pipeline will embed documents with DPR passage encoder into a FAISS index file|
|--toy_example|this is a "store_true" flag, you either include it or not in the command|if included, the pipeline will run a toy example with a small subset of images|


### Command line arguments related to databases

| Command line argument | What it is|Default value|
|-----------------------|-----------|--------------|
|--index_name|name of the index table to be stored in database(s)|dureadervis-documents|
|--index_file|path to the faiss index file in haystack-api container|/home/user/output/index_files/faiss-indexfile.faiss|
    
You can experiment with the following FAISS document store parameters to tune the DPR retrieval performance. The higher the values, the better recall and MRR, but the retrieval speed will get slower.
| Command line argument | What it is|Default value|
|-----------------------|-----------|--------------|
|--faiss_nlinks|n_links param for faiss document store|512|
|--faiss_efsearch|ef_search param for faiss document store|128|
|--faiss_efconstruct|ef_construct param for faiss document store|200|
  
    
### Image preprocessing
The following methods are supported through command line arguments in scripts/run_distributed_indexing .sh
|Method|What it does|command line argument|
|------|------------|---------------------|
|grayscale|turn a color image into gray-scale image| --preprocess grayscale|
|binarize|turn a color image into blank-white image with locally adaptive thresolding| --preprocess binarize|
|none|do nothing to the image, use the original| --preprocess none|
|crop_image|crop images to a certain size, you can customize the cropping in ```src/utils.py```|--crop_image|

### OCR engines
Currently we support the following two OCR engines, you can pick one through command line arguments in ```scripts/run_distributed_indexing .sh```
|OCR engine|What it is|command line argument with example values|
|----------|----------|---------------------|
|PaddleOCR|OCR engine developed by Baidu https://github.com/PaddlePaddle/PaddleOCR|--ocr_engine paddleocr|
|Tesseract|OCR engine developed by HP and Google https://github.com/tesseract-ocr/tesseract|--ocr_engine tesseract --ocr_lang chi_sim|

PaddleOCR and Tesseract both support multiple languages, please refer to their documentations on what languages are supported.
* For PaddleOCR: You need to change the ```lang``` arg in ```src/test_pocr.py``` for a language other than Chinese.
* For Tesseract: You need to install additional packages for Tesseract if you want to extract text in a language other than English.
</br>

Advanced users: If you want to use another OCR engine, you need to implement your own ocr method and change the source code of ```src/gen-sods-doc-image-ray.py``` accordingly.

### Post processing of OCR outputs
The following methods are supported through command line arguments in ```scripts/run_distributed_indexing.sh```:
1. Splitting </br>
This method splits document text into passages of certain length and certain overlap, each passage should be at least certain length long. </br>
For example: 
```--max_seq_len_passage 500 --overlap 10 --min_chars 5 --split_doc```
The ```--split_doc``` flag tells the pipeline to split the document, then the other args specify the splitting params: each passage max 500 string length with 10 string length overlap, minimum length of a passage is 5, any passages shorter than length of 5 will be discarded.

2. Clustering </br>
This method clusters each text blocks extracted by OCR engine into passages. The blocks are firstly embedded into vectors with a pretrained language model, and then principal component analysis (PCA) is done on the vectors to reduce dimensionality, and then Kmeans clustering is performed on the reduced vectors, then finally the blocks belonging to the same cluster are concatenated together to form a passage. There is an option to further split the clustered passages into shorter ones. There is also an option to force the number of clusters to be 2 or to dynamically determine the number of clusters based on the length of the documents. </br>

For example: 
```
--cluster_doc --cluster_model microsoft/infoxlm-base --max_seq_len_passage 500 
```
The command line arguments above will use the specified model (example here is microsoft/infoxlm-base) to embed the text blocks and dynamically decide the number of clusters based on the total length of the document and the specified max length of passages (for example 500 here). </br>
```
--cluster_doc --cluster_model microsoft/infoxlm-base --force_num_cluster
```
The command line arguments above will force the number of clusters to be 2. </br>

```
--cluster_doc --cluster_model microsoft/infoxlm-base --force_num_cluster --split_doc --max_seq_len_passage 500 --overlap 10 --min_chars 5
```
The command line arguments above will force the number of clusters to be 2, and then for each cluster it will split the text into passages of specified max length/overlap/min length. </br>



## Customize Deployment Pipeline 
### Docker-compose yaml file and env config file
In most cases, you don't need to change the docker-compose yaml files. You can pick from one of the three docker-compose yaml files we provided for the three retrieval methods: bm25, dpr, ensemble. </br>
The env config file may need editing. The variables are described below. In most cases, you just need to change the ```PIPELINE_PATH``` to your customized pipeline yaml file and leave the other variables unchanged as long as you follow the folder structure described [here](../README.md#getting-started). If you changed the index name, then you need to update the ```INDEX_NAME``` variable accordingly.
|Variable|What it does|Value|
|--------|------------|-------------|
|PIPELINE_NAME|functionality of the pipeline|fixed at 'query', no need to change|
|PIPELINE_PATH|path to pipeline yaml file (path inside haystack-api container)|Example: /home/user/application/configs/pipelines_ensemble.haystack-pipeline.yml|
|APPLICATION|volume to be mounted to haystack-api container that has the use case resource code|```<your-path-to-source-code>```|
|MODEL|volume to be mounted to haystack-api container that has the DPR models|```<your-path-to-dpr-models>```|
|ESDB|volume to be mounted to elasticsearch container that has the elasticsearch database files|```<your-path-to-esdb-files>```|
|DB|volume to be mounted to postgresql container that has the postgresql database files|```<your-path-to-postgresqldb-files>```|
|INDEX_NAME|name of the index table|must match the index name specified in indexing|



### Pipeline config file for haystack
You can use one of the 3 pipeline yaml files we provided as the starting point. The params that you can customize are listed in the table below.

|Retrieval method|Component|Param|What it is|
|----------------|---------|-----|----------|
|BM25, Ensemble|DocumentStore|index|the index name specified in indexing|
|DPR, Ensemble|DocumentStore|faiss_index_path|path to the faiss index file (.faiss file) in the haystack-api container|
|DPR, Ensemble|DensePassageRetriever|query_embedding_model, passage_embedding_model|path to the fine-tuned local DPR models or pretrained model names on Huggingface model hub|
|Ensemble|Ensembler|weights|weighting factors for BM25 and DPR retrievers, the first number is BM25 weight and the second number is DPR weight|
|BM25, DPR, Ensemble|Retriever|top_k|how many documents to retrieve by BM25 or DPR retrievers|
|Ensemble|Ensembler|top_k_join|how many documents to be returned by ensembler|


### UI config file
**Important**: 
1. Please do not change the filename of the UI config file, leave it as "config.yml".
2. Do not change the volume mount path in the UI container in the docker-compose yaml file. </br>

There are two parts in the UI config file:
1. dataset: only one dataset is allowed
2. pipelines: can have multiple pipelines

To customize the UI:
1. dataset: specify the name of your own dataset and some questions that you want to show as examples on the webpage.
2. pipelines: you can copy one of the pipelines in the UI config file, paste to the bottom of the file and modify as needed. For example, if you have a cutomized pipeline config yaml called "pipelines.Ranker.haystack-pipeline.yml" where you have a Retriever followed by a Ranker. The pipeline yaml file may look similar to the following:

```
components:
  - name: DocumentStore
    type: ElasticsearchDocumentStore
    params:
      host: localhost
      index: document
  - name: Retriever
    type: BM25Retriever
    params:
      document_store: DocumentStore
      top_k: 100
  - name: Ranker
    type: SentenceTransformersRanker
    params:
      model_name_or_path: /home/user/model/my_ranker_model
      top_k: 5
```
Then for the pipelines section in the UI config file, you can add a sub-section like the following to the bottom of the UI config file.
```
- name: pipelines.Ranker.haystack-pipeline.yml
    top_k_sliders:
      - name: answer
        desc: "Max. number of answers"
        default_value: 5
        keys:
          - key: Ranker # match component name in pipeline yaml
            param: top_k # match param name of Ranker in pipeline yaml
      
      - name: retriever
        desc: "Max. number of documents from retriever"
        default_value: 100
        keys:
          - key: Retriever # match component name in pipeline yaml
            param: top_k # match param name of Retriever in pipeline yaml
```
The most important variables are the ```keys```, the ```key``` must match the ```name``` of the component in the pipeline yaml file, and the ```param``` in UI config should match the ```param``` in pipeline yaml.










