# Document Automation Reference Use Case

This blueprint is a one click refkit to provide an end-to-end solution for building deploy Document Automation, an AI-augmented multi-modal semantic search system for enterprise document images (for example, scanned documents). 
Nowadays, enterprises usually accumulate a vast quantity of documents, a large portion of which is in image formats such as scanned documents. These documents contain a large amount of valuable information, but it is a challenge for enterprises to index, search and gain insights from the document images due to challenges listed as below:
* The end-to-end (e2e) solution involves many components that are not easy to integrate together.
* Indexing a large document image collection is very time consuming.
* Query against document image collections with natural language requires multi-modality AI that understands both images and languages. Building such multi-modality AI models requires deep expertise in machine learning.
* Deploying multi-modality AI models together with databases and a user interface is not an easy task.
* Majority of multi-modality AI models can only comprehend English, developing non-English models takes time and requires ML experience.

This solution implement and demonstrate a complete end-to-end solution that helps enterprises tackle the retrieval tasks for document visual collections, and provides new state-of-the-art (SOTA) retrieval recall & mean reciprocal rank (MRR) on the benchmark dataset.

## Flow
1. Click on `Use Blueprint` button.
2. You will be redirected to your blueprint flow page.
3. Go to the project settings section and update the configuration or leave as default to check on built-in demo. To change the `dir_url` in first task allows you to try with other datasources.The input data should contains two parts: the finetuning dataset and index data.
4. Click on the `Run Flow` button.
5. The system will automatically extract information from provided images, finetune on new information and provide as indexing service.
6. Expected output is `DPR models and FAISS index files`. DRP models are finetuned model based on provided documents. FAISS files are stored in a index_files folder. Besides, there will be tmp files generated by postgres and elasticsearch (more than 1300 files).

<div class="warning">

**NOTICE**
> Flow provided by default uses a sampled Chinese data from SOTA dataset called 'Dureader-vis'. To show case the capability of providing non-English multi-modality AI models with less effort from users.<br>
> For future customizing, please read more in this link: [https://github.com/intel/document-automation/tree/main#how-to-customize-this-reference-kit](https://github.com/intel/document-automation/tree/main#how-to-customize-this-reference-kit)

## Solution Technical Overview
The architecture of the reference use case is shown in the figure below. It is composed of 3 pipelines: 
* Single-node Dense Passage Retriever (DPR) fine tuning pipeline
* Image-to-document indexing pipeline (can be run on either single node or distributed on multiple nodes)
    * By default, the Postgres and Elasticsearch will be launched automatically in the container for indexing. In addition, developers can specify their own service by passing parameters. 
* Single-node deployment pipeline
</br>

![usecase-architecture](assets/usecase-architecture.PNG)

## Solution Technical Details
* **Developer productivity**: The 3 pipelines in this reference use case are all containerized and allow customization through either command line arguments or config files. Developers can bring their own data and jump start development very easily. 
* **New state-of-the-art (SOTA) retrieval recall & mean reciprocal rank (MRR) on the benchmark dataset**: better than the SOTA reported in [this paper](https://aclanthology.org/2022.findings-acl.105.pdf) on [Dureader-vis](https://github.com/baidu/DuReader/tree/master/DuReader-vis), the largest open-source document visual retrieval dataset (158k raw images in total). We demonstrated that AI-augmented ensemble retrieval method achieved higher recall and MRR than non-AI retrieval method (see the table below).
* **Performance**: distributed capability significantly accelerates the indexing process to shorten the development time.
* **Deployment made easy**: using two Docker containers from [Intel's open domain question answering workflow](https://github.com/intel/open-domain-question-and-answer) and two other open-source containers, you can easily deploy the retrieval solution by customizing the config files and running the launch script provided in this reference use case.
* **Multilingual customizable models**: you can use our pipelines to develop and deploy your own models in many different languages.

### Retrieval Performance of the Dev Set Queries on the Entire Indexed Dureader-vis Dataset
| Method | Top-5 Recall | Top-5 MRR | Top-10 Recall | Top-10 MRR |
|------|------------|---------|-------------|----------|
| BM25 only (ours) | 0.7665 | 0.6310 | 0.8333 | 0.6402 |
| DPR only (ours) | 0.6362 | 0.4997 | 0.7097 | 0.5094 |
| **Ensemble (ours)** | **0.7983** | **0.6715** | **0.8452** | **0.6780** |
| SOTA reported by Baidu | 0.7633 | did not report | 0.8180 | 0.6508 |

## Learn More
To read about other use cases and workflows examples, see these resources:
- [Document Automation Reference Use Case: End-to-End AI Augmented Semantic Search System](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Document-Automation-Reference-Use-Case-End-to-End-AI-Augmented/post/1526342)
- [Intel's Open Domain Question Answering workflow](https://github.com/intel/open-domain-question-and-answer)
- [Developer Catalog](https://developer.intel.com/aireferenceimplementations)
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
    
## Support
The [Document Automation Github Repo](https://github.com/intel/document-automation) tracks both bugs and enhancement requests using issues.
