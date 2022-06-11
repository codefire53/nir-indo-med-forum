from common import  (
    parse_corpus, add_bm25_score, add_lmd_score, add_classic_score, dot_product, rank_results
)
import argparse
import json
from tqdm import tqdm
import numpy as np
import numpy as np
import pickle


def add_dpr_score(results: list, pembeddings: str, qembeddings: str) -> list:
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

        for p_idx, p_row in enumerate(q_row['ctxs']):
            pid = p_row['id']
            p_vector = pembeddings_map[pid]

            results[q_idx]['ctxs'][p_idx]['total_score'] = dot_product(q_vector, p_vector)
    return results



    return precision, mrr, bpref, map_score
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpr_retrieval_result", type=str, default=None)
    parser.add_argument("--sparse_retrieval_result", type=str, default=None)
    parser.add_argument("--pembeddings", type=str, default=None)
    parser.add_argument("--qembeddings", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--sparse_type", default="bm25", choices=["bm25","classic", "lmd"])
    parser.add_argument("--fine_tune_data", type=str, default='tydi')

    args = parser.parse_args()

    if args.dpr_retrieval_result:
        assert  args.corpus is not None
        corpus = parse_corpus(args.corpus)
        with open(args.dpr_retrieval_result, 'r') as f:
            retrieval_results = json.load(f)
        if args.sparse_type=="bm25":
            rerank_results = add_bm25_score(retrieval_results, corpus)
        elif args.sparse_type=='lmd':
            rerank_results = add_lmd_score(retrieval_results, corpus)
        else:
            rerank_results = add_classic_score(retrieval_results, corpus)
        rerank_results = rank_results(rerank_results, args.topk)
        with open(f"retrieve-dpr_rerank-{args.sparse_type}_results_{args.fine_tune_data}.json", 'w') as f:
            json.dump(rerank_results, f)
    
    else:
        assert args.pembeddings is not None and args.qembeddings is not None
        with open(args.sparse_retrieval_result, 'r') as f:
            retrieval_results = json.load(f)
        rerank_results = add_dpr_score(retrieval_results, args.pembeddings, args.qembeddings)
        rerank_results = rank_results(rerank_results, args.topk)
        with open(f"retrieve-{args.sparse_type}_rerank-dpr_results_{args.fine_tune_data}.json", 'w') as f:
            json.dump(rerank_results, f)
        
if __name__ == '__main__':
    main()