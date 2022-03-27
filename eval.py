import pandas as pd
import argparse
import json
import csv
from tqdm import tqdm

def calc_retrieval_score(result, rel_map: dict, negrel_map: dict, score_name: str) -> float:
    if score_name == 'mrr':
        mrr = 0.0
        q_len = 0
        for q_row in result:
            rank = 1
            found = False
            for ctx in q_row['ctxs']:
                if ctx['id'] in rel_map[q_row['question']['id']]:
                    found = True
                    break
                rank += 1
            mrr += (1/rank) if found else 0
            q_len+=1
        mrr /= q_len
        return mrr
    elif score_name == 'bpref':
        mean_bpref = 0.0
        q_len = 0
        for q_row in result:
            R = 0.0
            negrel_cumsum = 0.0
            n_above_r = []
            for ctx in q_row['ctxs']:
                if ctx['id'] in rel_map[q_row['question']['id']]:
                    n_above_r.append(negrel_cumsum)
                    R += 1.0
                elif ctx['id'] in negrel_map[q_row['question']['id']]:
                    negrel_cumsum += 1.0

            bpref = 0.0
            for n in n_above_r:
                bpref += 1-(n/R) if R > 0 else 1
            bpref = bpref/R if R > 0 else 0.0
            mean_bpref += bpref
            q_len+=1
        mean_bpref /= q_len
        return mean_bpref
    elif score_name == 'map':
        map_score = 0.0
        q_len = 0
        for q_row in result:
            q_len += 1
            Rq = 0.0
            Dq = 0.0
            total_ap = 0.0
            for ctx in q_row['ctxs']:
                Dq += 1.0
                if ctx['id'] in rel_map[q_row['question']['id']]:
                    Rq += 1.0
                    total_ap += Rq/Dq
            total_ap = total_ap/Rq if Rq > 0 else 0.0
            map_score += total_ap
        map_score /= q_len
        return map_score
    elif score_name == 'recall':
        mean_recall = 0.0
        q_len = 0
        for q_row in result:
            q_len += 1
            R = len(rel_map[q_row['question']['id']])
            recall = 0.0
            for ctx in q_row['ctxs']:
                recall += 1 if ctx['id'] in rel_map[q_row['question']['id']] else 0
            recall = recall/R if R > 0 else 0
            mean_recall += recall
        mean_recall /= q_len
        return mean_recall
    else:
        mean_precision = 0.0
        q_len = 0
        for q_row in result:
            number_of_passages  = 0
            precision = 0.0
            for ctx in q_row['ctxs']:
                precision += 1 if ctx['id'] in rel_map[q_row['question']['id']] else 0
                number_of_passages += 1
            precision /= number_of_passages
            mean_precision += precision
            q_len += 1
        mean_precision /= q_len
        return mean_precision

def eval_results(actual_results: list, gold_results: list, statistical_test_file=None):
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

    #print the scores
    print("-------------------------------RETRIEVAL SCORE-------------------------------")
    print(f"Precision: {precision}")
    print(f"MRR: {mrr}")
    print(f"MAP: {map_score}")
    print(f"BPref: {bpref}")
    print("-----------------------------------------------------------------------------")


    if statistical_test_file:
        print("Create file for evaluation scores statistical significance test")
        columns = ["Precision", "MRR", "MAP", "BPref"]
        with open(statistical_test_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(columns)
            for i in tqdm(range(1, len(actual_results)+1)):
                precision = calc_retrieval_score(actual_results[i-1:i], rel_data_dict, negrel_data_dict, 'precision')
                mrr = calc_retrieval_score(actual_results[i-1:i], rel_data_dict, negrel_data_dict, 'mrr')
                bpref = calc_retrieval_score(actual_results[i-1:i], rel_data_dict, negrel_data_dict, 'bpref')
                map_score = calc_retrieval_score(actual_results[i-1:i], rel_data_dict, negrel_data_dict, 'map')
                writer.writerow([precision, mrr, map_score, bpref])
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual_result", type=str, required=True)
    parser.add_argument("--gold_result", type=str, required=True)
    parser.add_argument("--statistical_test_file", type=str, default=None)
    args = parser.parse_args()

    with open(args.actual_result, 'r') as f:
        actual_data = json.load(f)

    with open(args.gold_result, 'r') as f:
        gold_data = json.load(f)

    eval_results(actual_data, gold_data, args.statistical_test_file)
if __name__=='__main__':
    main()
