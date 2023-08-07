# Document Automation Blueprint

Document Automation Refkit is an end-to-end reference solution for building an AI-augmented multi-modal semantic search system for document images (for example, scanned documents). This solution can help enterprises gain more insights from their document archives more quickly and easily using natural language queries.

The blueprints implement and demonstrate a complete end-to-end solution that helps enterprises tackle the retrieval tasks for document visual collections, and provides new state-of-the-art (SOTA) retrieval recall & mean reciprocal rank (MRR) on the benchmark dataset.

## How to Use 

### [CNVRG IO FLows] 

> Note: you can experience the workflow step by step and view each step logs and results. 

* How to execute: Flows -> Document Automation -> Click ‘Run’.

* How to view results: Experiments -> Click and Check each experiment result.


* Step-by-step explanation: 

   1) preprocess: seamlessly process files from Dureader-vis that contain questions, answers and document text extracted by PaddleOCR, and then split the document text into passages with max_length=500, overlap=10, min_length=5 by default, and identify the positive passage for each question.

    2) dpr fine-tuning: fine-tune DPR model with pretrained query_encoder and passage_encoder

    3) indexing: write text passages produced by the preprocessing task into one or two databases depending on the retrieval method that the user specified.
