# Dense Passage Retriever for Indonesian Consumer Health Similar Questions Retrieval

This repository all source codes for DPR application on Indonesian Consumer Health Similar Questions Retrieval which is my undergraduate thesis research.  This DPR uses IndoBERT as the base encoder which is trained across several ad-doc retrieval training datasets.

## Setup

1. Clone this repository
2. If you use virtualenv, create new virtual environment
   ```
   virtualenv .env
   ```
3. Install some dependencies needed to run this project
   ```
   pip install -r requirements.txt
4. Install pylucene. You can find the instruction how to install it ![here](https://lucene.apache.org/pylucene/install.html).

## Directories Organization

1. Analysis: Contains several notebooks for analysis like statistical testing, error analysis, etc.
2. Baseline: Sparse retriever model implementations
3. DPR: DPR and its variations implementations
4. Reranker: monoBERT implementation

## Files Organization

1. eval.py: Score evaluation 
2. preprocess.ipynb: Data preprocessing notebook

## Additional Resources

TBA