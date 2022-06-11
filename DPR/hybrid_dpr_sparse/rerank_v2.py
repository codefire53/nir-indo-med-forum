from eval import calc_retrieval_score
from common import  (
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
            results[q_idx]['ctxs'][p_idx]['dpr_score'] = dpr_score
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
    parser.add_argument("--fine_tune_data", type=str, default='tydi')
    args = parser.parse_args()
    '''
    #Tydi
    #These values were obtained from logistic regression. See param-tuning-logistic.ipynb
    best_alpha_bm25 = 1.2009257529289656
    best_beta_bm25 = 0.9245081251599654

    best_alpha_lmd = 0.7730687890211472
    best_beta_lmd = 0.6092880000015407

    best_alpha_classic = 0.9960027480830675
    best_beta_classic = 1.6228190542516596
    '''

    '''
    #MFAQ
    #These values were obtained from logistic regression. See param-tuning-logistic.ipynb
    best_alpha_bm25 = 1.2285416540664982
    best_beta_bm25 = 0.22739428333486594

    best_alpha_lmd = 0.7546587421837782
    best_beta_lmd =  -0.005160645636580339


    best_alpha_classic = 1.0834093187864764
    best_beta_classic = 0.25778051351383674
    '''

    '''
    #TTHealth
    #These values were obtained from logistic regression. See param-tuning-logistic.ipynb
    best_alpha_bm25 = 1.1592463186937745
    best_beta_bm25 = 1.3273780871051826

    best_alpha_lmd = 0.7069047663955942
    best_beta_lmd =  1.1833258679652499

    best_alpha_classic = 1.0219408455134382
    best_beta_classic = 1.5415919896199344
    '''
    
    '''
    #Indosum
    #These values were obtained from logistic regression. See param-tuning-logistic.ipynb
    best_alpha_bm25 = 1.169561009392178
    best_beta_bm25 = 0.5405541305894864


    best_alpha_lmd = 0.6822521161017481
    best_beta_lmd =  0.7469757689268173

    best_alpha_classic = 1.0532426034251743
    best_beta_classic = 0.999061265666952
    '''
    
    '''
    #TTMeqSum
    #These values were obtained from logistic regression. See param-tuning-logistic.ipynb
    best_alpha_bm25 = 1.1661759689477187
    best_beta_bm25 = 0.9573643744499755

    best_alpha_lmd = 0.7723904885182303
    best_beta_lmd =  0.727392291680184

    best_alpha_classic = 1.0577373223374467
    best_beta_classic = 1.680994034617557
    '''

    '''
    #Indowiki
    #These values were obtained from logistic regression. See param-tuning-logistic.ipynb
    best_alpha_bm25 = 1.238712038418987
    best_beta_bm25 = 0.24638944737423446

    best_alpha_lmd = 0.74106113133055
    best_beta_lmd =  0.4528767928434931


    best_alpha_classic = 1.1035365998705648
    best_beta_classic = 0.24618084040394633
    '''
    
    '''
    #ICT
    #These values were obtained from logistic regression. See param-tuning-logistic.ipynb
    best_alpha_bm25 = 1.2230756767092634
    best_beta_bm25 = -0.05387791400467227

    best_alpha_lmd = 0.7551215312790107
    best_beta_lmd = -0.045555025778324376

    best_alpha_classic = 1.1132625691974294
    best_beta_classic = 0.03142309027366281
    '''

    '''
    #Other
    #These values were obtained from logistic regression. See param-tuning-logistic.ipynb
    best_alpha_bm25 = 1.236989440226168
    best_beta_bm25 = 0.22634348798847928

    best_alpha_lmd = 0.779252917227329
    best_beta_lmd = 0.29605572928611174

    best_alpha_classic = 1.0537011739558793
    best_beta_classic = 0.048971990901022035
    '''
    corpus = parse_corpus(args.corpus)

    if args.bm25_query_result:
        print("Evaluate BM25-DPR")
        combined_query_results = join_results(args.dpr_query_result, args.bm25_query_result)
        combined_query_results = add_bm25_score(combined_query_results, corpus)
        query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, best_alpha_bm25, best_beta_bm25)
        query_results = rank_results(query_results, args.topk)
        with open(f"dpr-bm25-v2_query_results_{args.fine_tune_data}.json", 'w') as f:
            json.dump(query_results, f)

    if args.lmd_query_result:
        print("Evaluate LMDirichlet-DPR")
        combined_query_results = join_results(args.dpr_query_result, args.lmd_query_result)
        combined_query_results = add_lmd_score(combined_query_results, corpus)
        query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, best_alpha_lmd, best_beta_lmd)
        query_results = rank_results(query_results, args.topk)
        with open(f"dpr-lmd-v2_query_results_{args.fine_tune_data}.json", 'w') as f:
            json.dump(query_results, f)

    if args.classic_query_result:
        print("Evaluate TFIDF-DPR")
        combined_query_results = join_results(args.dpr_query_result, args.classic_query_result)
        combined_query_results = add_classic_score(combined_query_results, corpus)
        query_results = add_dpr_score(combined_query_results, args.pembeddings, args.qembeddings, best_alpha_classic, best_beta_classic)
        query_results = rank_results(query_results, args.topk)
        with open(f"dpr-classic-v2_query_results_{args.fine_tune_data}.json", 'w') as f:
            json.dump(query_results, f)

if __name__ == '__main__':
    main()