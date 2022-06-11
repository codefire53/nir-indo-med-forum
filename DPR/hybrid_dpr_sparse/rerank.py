
import copy
from common import (
    parse_corpus, join_results, add_bm25_score, add_classic_score, 
    add_lmd_score, rank_results, dot_product
)
import argparse
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
from typing import List, Tuple, Dict, Iterator
import sys
sys.path.append('../../')
from eval import calc_retrieval_score

def add_dpr_score(results: list, pembeddings: str, qembeddings: str, alpha: float) -> list:
    pembeddings_map = dict()
    with open(pembeddings, 'rb') as f:
        while True:
            try:
                passages_lst = pickle.load(f)
                for pid, p_vector in passages_lst:
                    pembeddings_map[pid] = np.array(p_vector)
            except EOFError:
                break
    
    qembeddings_map = dict()
    with open(qembeddings, 'rb') as f:
        while True:
            try:
                queries_lst = pickle.load(f)
                for qid, q_vector in queries_lst:
                    qembeddings_map[qid] = np.array(q_vector)
            except EOFError:
                break

    for q_idx, q_row in tqdm(enumerate(results)):
        qid = q_row['question']['id']
        q_vector = qembeddings_map[qid]
        max_dpr_score = -float('inf')
        min_dpr_score = float('inf')
        for p_idx, p_row in enumerate(q_row['ctxs']):
            pid = p_row['id']
            p_vector = pembeddings_map[pid]
            score = dot_product(q_vector, p_vector)
            max_dpr_score = max(score, max_dpr_score)
            min_dpr_score = min(score, min_dpr_score)

        for p_idx, p_row in enumerate(q_row['ctxs']):
            pid = p_row['id']
            p_vector = pembeddings_map[pid]
            dpr_score = (dot_product(q_vector, p_vector)-min_dpr_score)/max_dpr_score
            results[q_idx]['ctxs'][p_idx]['dpr_score'] =  dpr_score
            results[q_idx]['ctxs'][p_idx]['total_score'] =  (1-alpha)*results[q_idx]['ctxs'][p_idx]['sparse_score']+alpha*dpr_score
    return results


def get_eval_scores(actual_results: list, gold_results: list) -> Tuple[float, float, float, float]:
    #retrieve ground truth
    rel_data_dict = dict()
    negrel_data_dict = dict()
    for row in gold_results:
        rel_data_dict[row['question']['id']] = []
        negrel_data_dict[row['question']['id']] = []
        for ctx in row['ctxs']:
            rel_data_dict[row['question']['id']].append(ctx['id'])
        for ctx in row['neg_ctxs']:
            negrel_data_dict[row['question']['id']].append(ctx['id'])

    
    #gather scores
    precision = calc_retrieval_score(actual_results, rel_data_dict, negrel_data_dict, 'precision')
    mrr = calc_retrieval_score(actual_results, rel_data_dict, negrel_data_dict, 'mrr')
    bpref = calc_retrieval_score(actual_results, rel_data_dict, negrel_data_dict, 'bpref')
    map_score = calc_retrieval_score(actual_results, rel_data_dict, negrel_data_dict, 'map')

    return precision, mrr, bpref, map_score
def main():
    '''We use BPref as the main metric for hyperparameter tuning becasue the data is incomplete'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpr_query_result", type=str, required=True)
    parser.add_argument("--bm25_query_result", type=str, default=None)
    parser.add_argument("--lmd_query_result", type=str, default=None)
    parser.add_argument("--classic_query_result", type=str, default=None)
    parser.add_argument("--pembeddings", type=str, required=True)
    parser.add_argument("--qembeddings", type=str, required=True)
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--fine_tune_data", type=str, default='tydi')
    args = parser.parse_args()
    best_bm25_alpha = 0.9
    best_lmd_alpha = 0.1
    best_tfidf_alpha = 0.1
    alphas = [alpha/10 for alpha in range(1, 10)]
    with open(args.eval_file, 'r') as f:
        gold_results = json.load(f)
    corpus = parse_corpus(args.corpus)
    scores_dict = {
        'precision': [],
        'mrr': [],
        'bpref': [],
        'map': []
    }
    all_scores = {
        'bm25': copy.deepcopy(scores_dict),
        'lmd': copy.deepcopy(scores_dict),
        'tfidf': copy.deepcopy(scores_dict)
    }

    if args.bm25_query_result:
        print("Evaluate BM25-DPR")
        combined_query_results = join_results(args.dpr_query_result, args.bm25_query_result)
        combined_query_results = add_bm25_score(combined_query_results, corpus)
        #Hyperparameter tuning
        best_score = -float('inf')
        for alpha in alphas:

            query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, alpha)
            query_results = rank_results(query_results, args.topk)
            precision, mrr, bpref, map_score = get_eval_scores(query_results, gold_results)
            all_scores['bm25']['precision'].append(precision)
            all_scores['bm25']['mrr'].append(mrr)
            all_scores['bm25']['map'].append(map_score)
            all_scores['bm25']['bpref'].append(bpref)
            if bpref > best_score:
                best_score = bpref
                best_bm25_alpha = alpha
        assert len(all_scores['bm25']['precision']) == len(alphas)

        query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, best_bm25_alpha)
        query_results = rank_results(query_results, args.topk)
        with open(f"dpr-bm25_query_results_{args.fine_tune_data}.json", 'w') as f:
            json.dump(query_results, f)
        print(f"Best BM25 alpha: {best_bm25_alpha}")

    if args.lmd_query_result:
        print("Evaluate LMDirichlet-DPR")
        combined_query_results = join_results(args.dpr_query_result, args.lmd_query_result)
        combined_query_results = add_lmd_score(combined_query_results, corpus)

        #hyperparameter tuning
        best_score = -float('inf')
        for alpha in alphas:
            query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, alpha)
            query_results = rank_results(query_results, args.topk)
            precision, mrr, bpref, map_score = get_eval_scores(query_results, gold_results)
            all_scores['lmd']['precision'].append(precision)
            all_scores['lmd']['mrr'].append(mrr)
            all_scores['lmd']['map'].append(map_score)
            all_scores['lmd']['bpref'].append(bpref)
            if bpref > best_score:
                best_score = bpref
                best_lmd_alpha = alpha
        assert len(all_scores['lmd']['precision']) == len(alphas)

        query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, best_lmd_alpha)
        query_results = rank_results(query_results, args.topk)
        with open(f"dpr-lmd_query_results_{args.fine_tune_data}.json", 'w') as f:
            json.dump(query_results, f)
        print(f"Best LMDirichlet alpha: {best_lmd_alpha}")

    if args.classic_query_result:
        print("Evaluate TFIDF-DPR")
        combined_query_results = join_results(args.dpr_query_result, args.classic_query_result)
        combined_query_results = add_classic_score(combined_query_results, corpus)

        #hyperparameter tuning
        best_score = -float('inf')
        for alpha in alphas:
            query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, alpha)
            query_results = rank_results(query_results, args.topk)
            precision, mrr, bpref, map_score = get_eval_scores(combined_query_results, gold_results)
            all_scores['tfidf']['precision'].append(precision)
            all_scores['tfidf']['mrr'].append(mrr)
            all_scores['tfidf']['map'].append(map_score)
            all_scores['tfidf']['bpref'].append(bpref)
            if bpref > best_score:
                best_score = bpref
                best_tfidf_alpha = alpha
        assert len(all_scores['tfidf']['precision']) == len(alphas)
        print(best_tfidf_alpha)
        query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, best_tfidf_alpha)
        query_results = rank_results(query_results, args.topk)
        with open(f"dpr-classic_query_results_{args.fine_tune_data}.json", 'w') as f:
            json.dump(query_results, f)
        print(f"Best TFIDF alpha: {best_tfidf_alpha}")
    fig, axs = plt.subplots(len(all_scores.keys())+1, len(all_scores.values())+1)
    r = 0
    for model, model_val in all_scores.items():
        c = 0
        for scoring, values in model_val.items():
            axs[c,r].plot(alphas, values)
            axs[c,r].set_xlabel("alpha")
            axs[c,r].set_ylabel(f"{scoring} score")
            axs[c,r].set_title(model)
            c+=1
        r += 1
    fig.tight_layout()
    fig.set_size_inches(10, 10)
    fig.savefig('alphas.png', dpi=100)
if __name__ == '__main__':
    main()