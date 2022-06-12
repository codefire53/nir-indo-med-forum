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
4. Install pylucene. You can find the instruction how to install it [here](https://lucene.apache.org/pylucene/install.html).

## Directories Organization

1. Analysis: Contains several notebooks for analysis like statistical testing, error analysis, etc.
2. Baseline: Sparse retriever model implementations
3. DPR: DPR and its variations implementations
4. Reranker: monoBERT implementation

## Files Organization

1. eval.py: Score evaluation. Also can produce score per query that can be used for statistical significance test by supplying --statistical_test_file param.
2. preprocess.ipynb: Data preprocessing notebook

## Additional Resources

Non-source code resources such as datasets and model checkpoints that were used for this research can be accessed [here](https://univindonesia-my.sharepoint.com/personal/mahardika_krisna_office_ui_ac_id/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fmahardika%5Fkrisna%5Foffice%5Fui%5Fac%5Fid%2FDocuments%2FDPR%20Indonesian%20Consumer%20Health%20Similar%20Questions%20Retrieval%2FDatas%2FTraining%20Datas%2FTTHealth%2FRaw%20Data).