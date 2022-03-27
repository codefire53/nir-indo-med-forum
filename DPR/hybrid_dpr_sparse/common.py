import argparse
import json
import csv
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from operator import itemgetter
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
import pickle
from typing import List, Tuple, Dict, Iterator

def parse_corpus(corpus: str) -> dict:
    doc_map = dict()
    with open(corpus, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            full_text = row[2] if len(row) == 2 and row[2] is not None else ""
            if full_text == "":
                full_text = row[1]
            else:
                full_text += ". "+row[1]
            doc_map[row[0]] = full_text
    return doc_map

def join_results(dpr_result: str, sparse_result: str) -> list:
    with open(dpr_result, 'r') as f:
        dpr_json = json.load(f)
    with open(sparse_result, 'r') as f:
        sparse_json = json.load(f)
    combined_data = []
    for idx, row in enumerate(dpr_json):
        pos = idx
        while sparse_json[pos]['question']['id'] !=  dpr_json[idx]['question']['id']:
            pos+=1
        row_map = dict()
        row_map['question'] = row['question']
        row_map['ctxs'] = row['ctxs'] 
        existing_id = [ctx['id'] for ctx in row['ctxs']]

        sparse_ctx_row = [row for row in sparse_json[pos]['ctxs'] if row['id'] not in existing_id]
        row_map['ctxs'] += sparse_ctx_row
        combined_data.append(row_map)
    return combined_data



def bm25_score(doc: str, word_freq: int, N_q: int, avgdl: float, N: int, k1: float, b: float) -> float:
    #tf
    tf =  (word_freq*(k1+1))/(word_freq + k1 * (1 - b + b * len(doc) / avgdl))\
    #idf
    idf = np.log(((N - N_q + 0.5) / (N_q + 0.5)) + 1)
    return round(tf*idf, 4)

def bm25_init(docs_dict: dict) ->Tuple[dict, dict, int, float]:
    doc_maps = defaultdict(int)
    term_maps = dict()
    
    for doc_id, doc_text in docs_dict.items():
        doc_words = list(word_tokenize(doc_text))
        doc_terms_map = defaultdict(int)
        words_exist = defaultdict(bool)
        for word in doc_words:
            if not words_exist[word]:
                doc_maps[word] += 1 
                words_exist[word] = True
                
            doc_terms_map[word] += 1
        term_maps[doc_id] = doc_terms_map
    N = len(list(docs_dict.keys()))
    avgdl = sum(len(doc) for doc in list(docs_dict.values())) / len(list(docs_dict.values()))
    return doc_maps, term_maps, N, avgdl

def bm25_query_score(query: str, doc_id: str, docs_freq_map: dict, terms_freq_map: dict, docs_map: dict, avgdl: float, N: int, k1: float, b:float) -> float:
    query = list(word_tokenize(query))
    selected_doc = docs_map[doc_id]
    res = sum([bm25_score(selected_doc, terms_freq_map[doc_id][word], docs_freq_map[word], avgdl, N, k1, b) for word in query])
    return res


def add_bm25_score(results: list, corpus: dict, k1=1.2, b=0.75) -> list:
    passages_freq_map, terms_freq_map, N, avgdl = bm25_init(corpus)
    for q_idx, q_row in tqdm(enumerate(results)):
        query = q_row['question']['title']
        query += f". {q_row['question']['text']}" if q_row['question']['text'] != "" else ""
        max_bm25_score = -float('inf')
        min_bm25_score = float('inf')
        for p_idx, p_row in enumerate(q_row['ctxs']):
            pid = p_row['id']
            score = bm25_query_score(query, pid, passages_freq_map, terms_freq_map, corpus, avgdl, N, k1, b)
            max_bm25_score = max(max_bm25_score, score)
            min_bm25_score = min(min_bm25_score, score)
        for p_idx, p_row in enumerate(q_row['ctxs']):
            pid = p_row['id']
            results[q_idx]['ctxs'][p_idx]['bm25_score'] = (bm25_query_score(query, pid, passages_freq_map, terms_freq_map, corpus, avgdl, N, k1, b)-min_bm25_score)/(max_bm25_score-min_bm25_score)
    return results

def lmd_init(docs_dict: dict) -> Tuple[dict, dict, dict, float]:
    term_maps = dict()
    total_term_maps = defaultdict(int)
    total_words = 0
    collection_maps = defaultdict(int)
    for doc_id, doc_text in docs_dict.items():
        doc_words = list(word_tokenize(doc_text))
        doc_terms_map = defaultdict(int)
        for word in doc_words:
            total_words += 1
            collection_maps[word] += 1
            doc_terms_map[word] += 1
            total_term_maps[doc_id] += 1
        term_maps[doc_id] = doc_terms_map
    return term_maps, total_term_maps, total_words, collection_maps

def lmd_score(word_freq_in_doc: int, total_doc_words: int, word_freq_in_corpus: int, total_corpus_words: int, miu: float) -> float:
    numerator = word_freq_in_doc + miu*(word_freq_in_corpus/total_corpus_words)
    denumerator = total_doc_words + miu
    return numerator/denumerator

def log_or_zero(value: float) -> float:
    return np.log(value) if value > 0 else 0.0

def lmd_query_score(query: str, doc_id: str, terms_freq_map: dict, total_terms_freq_map: dict, total_collection_words: int, collection_maps: dict, miu:float) -> float:
    query = list(word_tokenize(query))
    res = sum([log_or_zero(lmd_score(terms_freq_map[doc_id][word], total_terms_freq_map[doc_id], collection_maps[word], total_collection_words, miu)) for word in query])
    return res

def add_lmd_score(results: list, corpus: dict, miu=2000) -> list:
    term_maps, total_term_maps, total_words, collection_maps = lmd_init(corpus)
    for q_idx, q_row in tqdm(enumerate(results)):
        query = q_row['question']['title']
        query += f". {q_row['question']['text']}" if q_row['question']['text'] != "" else ""
        max_lmd_score = -float('inf')
        min_lmd_score = float('inf')
        for p_idx, p_row in enumerate(q_row['ctxs']):
            pid = p_row['id']
            score = lmd_query_score(query, pid, term_maps, total_term_maps, total_words, collection_maps, miu)
            max_lmd_score = max(max_lmd_score, score)
            min_lmd_score = min(min_lmd_score, score)
        for p_idx, p_row in enumerate(q_row['ctxs']):
            pid = p_row['id']
            results[q_idx]['ctxs'][p_idx]['lmd_score'] = (lmd_query_score(query, pid, term_maps, total_term_maps, total_words, collection_maps, miu)-min_lmd_score)/(max_lmd_score-min_lmd_score)
    return results

def dot_product(x, y) -> np.float64:
    return np.float64(np.dot(x, y.T))
    
def cosine_similarity(x, y) ->np. float64:
    return dot_product(x, y)/np.float64((np.linalg.norm(x)*np.linalg.norm(y)))

def classic_init(docs_dict: dict) ->Tuple[dict, dict, int, list, dict]:
    doc_maps = defaultdict(int)
    term_maps = dict()
    terms_list = set()
    term2id = dict()
    for doc_id, doc_text in tqdm(docs_dict.items()):
        doc_words = list(word_tokenize(doc_text))
        doc_terms_map = defaultdict(int)
        words_exist = defaultdict(bool)
        for word in doc_words:
            if not words_exist[word]:
                doc_maps[word] += 1 
                words_exist[word] = True
            terms_list.add(word)
            doc_terms_map[word] += 1
        term_maps[doc_id] = doc_terms_map
    N = len(list(docs_dict.keys()))
    terms_list = list(terms_list)
    term2id = {term:idx for idx,term in enumerate(terms_list)}
    return doc_maps, term_maps, N, terms_list, term2id

def classic_score(word_freq: int, N_q: int, N: int) -> float:
    tf = word_freq
    idf = np.log(((N - N_q + 0.5) / (N_q + 0.5)) + 1)
    return round(tf*idf, 4)

def classic_query_score(query: str, doc: str, doc_id: str, docs_freq_map: dict, terms_freq_map: dict, N: int, terms_list: List[str], term2idx: dict, doc_vectors: dict) -> float:
    query = list(word_tokenize(query))
    doc = list(word_tokenize(doc))
    dim = len(terms_list)
    query_vector = np.zeros((1, dim))
    doc_vector = np.zeros((1, dim))
    if doc_id not in doc_vectors:
        for word in doc:
            if word not in terms_list:
                continue
            doc_vector[0][term2idx[word]] += classic_score(terms_freq_map[doc_id][word],  docs_freq_map[word], N)
        doc_vectors[doc_id] = doc_vector
    else:
        doc_vector = doc_vectors[doc_id]
    for word in query:
        if word not in terms_list:
            continue
        query_vector[0][term2idx[word]] += classic_score(terms_freq_map[doc_id][word],  docs_freq_map[word], N)

    res = cosine_similarity(query_vector, doc_vector)
    return res

def add_classic_score(results: list, corpus: dict) -> list:
    passages_freq_map,terms_freq_map, N, terms_list, term2idx = classic_init(corpus)
    doc_vectors = defaultdict()
    for q_idx, q_row in tqdm(enumerate(results)):
        query = q_row['question']['title']
        query += f". {q_row['question']['text']}" if q_row['question']['text'] != "" else ""
        max_classic_score = -float('inf')
        min_classic_score = float('inf')
        for p_idx, p_row in enumerate(q_row['ctxs']):
            pid = p_row['id']
            doc = p_row['title']
            doc += f". {p_row['text']}" if p_row['text'] != "" else ""
            score = classic_query_score(query, doc, pid, passages_freq_map, terms_freq_map, N, terms_list, term2idx, doc_vectors)
            max_classic_score = max(max_classic_score, score)
            min_classic_score = min(min_classic_score, score)
        for p_idx, p_row in enumerate(q_row['ctxs']):
            pid = p_row['id']
            doc = p_row['title']
            doc += f". {p_row['text']}" if p_row['text'] != "" else ""
            results[q_idx]['ctxs'][p_idx]['classic_score'] = (classic_query_score(query, doc, pid, passages_freq_map, terms_freq_map, N, terms_list, term2idx, doc_vectors)-min_classic_score)/(max_classic_score-min_classic_score)
    print(doc_vectors)
    return results

def rank_results(results: list, topk: int) -> list:
    for q_idx, q_row in tqdm(enumerate(results)):
        results[q_idx]['ctxs'] = sorted(results[q_idx]['ctxs'], key=itemgetter('total_score'), reverse=True)[:topk]
    return results