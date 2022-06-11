# Sparse Retriever Model
This directory contains source code for sparse retriever models which is implemented using PyLucene. 

## How to run
   ```
   python lucene_searcher.py --corpus_path <insert-your-corpus-file-here> --query_path <insert-your-query-file-here> --topk <top-k-documents-for-each-query>
   ```
For example. Lets say we want to document retrieval from corpus.tsv using queries from test-query.json and return top-10 relevants for each query.
   ```
   python lucene_searcher.py --corpus_path corpus.tsv --query_path test-query.json --topk 10
   ```

## Corpus File
Corpus file can be on either tsv format or json format. For tsv format, it should be formatted like this for each line.
```
<document-id> <document-text> <document-title>
```
For json format, it should be formatted like this.
```
    [
        {
            'docid': "document-id",
            'text': "document-text",
            'title': "document-title"
        },
        ...
    ]
```


## Query File
Query should be a json file with this following format
```
    [
        {
            'question': {
                ...
                'title': "question-title",
                'text': "question-text"
            }
        },
        ...
    ]
```