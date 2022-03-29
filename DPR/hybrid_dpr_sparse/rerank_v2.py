from eval import calc_retrieval_score
from hybrid_dpr_sparse.common import  (
    parse_corpus, join_results, add_bm25_score, add_lmd_score, add_classic_score, dot_product, rank_results
)
import argparse
import json
from tqdm import tqdm
import numpy as np
import numpy as np
import pickle

def add_dpr_score(results: list, pembeddings: str, qembeddings: str, alpha: float, beta: float) -> list:
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
            results[q_idx]['ctxs'][p_idx]['total_score'] = alpha*results[q_idx]['ctxs'][p_idx]['total_score']+beta*dpr_score
    return results

def main():
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
    args = parser.parse_args()

    #These values are obtained from logistic regression. See param-tuning-logistic.ipynb
    best_alpha_bm25 = 1.2025570627685844
    best_beta_bm25 = 0.9407036390334909

    best_alpha_lmd = 0.77809029740458
    best_beta_lmd = 0.5507165531926917

    best_alpha_classic = 0.9685231942879037
    best_beta_classic = 1.6053453674871936

    corpus = parse_corpus(args.corpus)

    if args.bm25_query_result:
        print("Evaluate BM25-DPR")
        combined_query_results = join_results(args.dpr_query_result, args.bm25_query_result)
        combined_query_results = add_bm25_score(combined_query_results, corpus)
        query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, best_alpha_bm25, best_beta_bm25)
        query_results = rank_results(query_results, args.topk)
        with open("dpr-bm25-v2_query_results.json", 'w') as f:
            json.dump(query_results, f)

    if args.lmd_query_result:
        print("Evaluate LMDirichlet-DPR")
        combined_query_results = join_results(args.dpr_query_result, args.lmd_query_result)
        combined_query_results = add_lmd_score(combined_query_results, corpus)
        query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, best_alpha_lmd, best_beta_lmd)
        query_results = rank_results(query_results, args.topk)
        with open("dpr-lmd-v2_query_results.json", 'w') as f:
            json.dump(query_results, f)

    if args.classic_query_result:
        print("Evaluate TFIDF-DPR")
        combined_query_results = join_results(args.dpr_query_result, args.classic_query_result)
        combined_query_results = add_classic_score(combined_query_results, corpus)
        query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, best_alpha_classic, best_beta_classic)
        query_results = rank_results(query_results, args.topk)
        with open("dpr-classic-v2_query_results.json", 'w') as f:
            json.dump(query_results, f)

if __name__ == '__main__':
    main()