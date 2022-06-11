#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
import jsonlines
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, index: DenseIndexer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str], embed_title: bool = False) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):
                if not embed_title:
                    batch_token_tensors = [self.tensorizer.text_to_tensor(q[0]) for q in questions[batch_start:batch_start + bsz]]
                else:
                    batch_token_tensors = [self.tensorizer.text_to_tensor(q[0], q[1]) for q in questions[batch_start:batch_start + bsz][0]]
                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def index_encoded_data(self, vector_files: List[str], buffer_size: int = 50000):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files)):
            db_id, doc_vector = item
            buffer.append((db_id, doc_vector))
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info('Data indexing completed.')

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info('index search time: %f sec.', time.time() - time0)
        return results


def parse_qa_tsv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers

def read_json(file_name):
    json_data = []
    print("loading examples from {0}".format(file_name))
    with open(file_name, 'r') as json_out:
        json_data = json.load(json_out)
    return json_data

def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

def parse_qa_json_file(location) -> Iterator[Tuple[Tuple[str, str], str]]:
    data = read_json(location)
    for row in data:
        question = (row["question"]["text"], row["question"]["title"])
        q_id = row["question"]["id"]
        yield question, q_id

def parse_qa_jsonlines_file(location) -> Iterator[Tuple[Tuple[str, str], str]]:
    data = read_jsonlines(location)
    for row in data[0]:
        question = (row["question"]["text"], row["question"]["title"])
        q_id = row["question"]["id"]
        yield question, q_id


def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info('Reading data from: %s', ctx_file)
    if ctx_file.startswith(".gz"):
        with gzip.open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    
    elif ctx_file.endswith('.json'):
        try:
            with open(ctx_file , 'r') as f:
                corpus_data = json.load(f)
        except UnicodeDecodeError:
            with open(ctx_file, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
        for row in corpus_data:
            docs[row['docid']] = (row['text'], row['title'])
    
    elif ctx_file.endswith('.jsonl'):
        try:
            with open(ctx_file, 'r') as f:
                corpus_data_lst = list(f)
        except UnicodeDecodeError:
            with open(ctx_file, 'r', encoding='utf-8') as f:
                corpus_data_lst = list(f)
        for json_str in corpus_data_lst:
            json_dict = json.loads(json_str)
            docs[json_dict['docid']] = (json_dict['text'], json_dict['title'])

    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    return docs

def save_results(passages: Dict[object, Tuple[str, str]], questions: List[Tuple[str, str]], q_ids: List[str], 
    top_passages_and_scores: List[Tuple[List[object], List[float]]], out_file: str, out_file_prettified: str):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    for i, q in enumerate(questions):
        q_id = q_ids[i]
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        # q format: (text, title)
        merged_data.append({
            'question': {
                'id': q_id,
                'title': q[1], 
                'text': q[0]
            },
            'ctxs': [
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c]
                } for c in range(ctxs_num)
            ]
        })

    with open(out_file, "w") as writer:
        json.dump(merged_data, writer)

    with open(out_file_prettified, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info('Saved results * scores  to %s(main file) & %s(prettified file)', out_file, out_file_prettified)

def expand_question(text1, text2):
    tokenized_text1 = list(word_tokenize(text1))[:]
    tokenized_text2 = list(word_tokenize(text2))[:]
    final_tokenized_text = tokenized_text1 + tokenized_text2
    return " ".join(final_tokenized_text)

def expand_questions(questions: List[Tuple[str, str]], top_passages_and_scores: List[Tuple[List[object], List[float]]], passages: Dict[object, Tuple[str, str]]) -> List[Tuple[str, str]]:
    for  i, q in enumerate(questions):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        choosen_doc =  docs[0]
        doc_text = choosen_doc[0]
        doc_title = choosen_doc[1]
        questions[i] = (f"{expand_question(q[0],doc_text)}", f"{expand_question(q[1], doc_title)}")
    return questions


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info('Reading file %s', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def main(args):
    if args.model_file:
        saved_state = load_states_from_checkpoint(args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')

    if args.model_file:
        prefix_len = len('question_model.')
        question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                                key.startswith('question_model.')}
        model_to_load.load_state_dict(question_encoder_state)
    vector_size = model_to_load.get_out_size()
    logger.info('Encoder vector_size=%d', vector_size)

    index_buffer_sz = args.index_buffer
    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size)
        index_buffer_sz = -1  # encode all at once
    else:
        index = DenseFlatIndexer(vector_size)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)


    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)
    if args.save_or_load_index and os.path.exists(index_path):
        retriever.index.deserialize(index_path)
    else:
        logger.info('Reading all passages data from files: %s', input_paths)
        retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz)
        if args.save_or_load_index:
            retriever.index.serialize(index_path)
    # get questions
    questions = []
    q_ids = []

    for ds_item in parse_qa_json_file(args.qa_file):
        question, q_id = ds_item
        questions.append(question)
        q_ids.append(q_id)

    questions_tensor = retriever.generate_question_vectors(questions)
    
    all_passages = load_passages(args.ctx_file)

    if args.use_pseudo_feedback:
        top_ids_and_scores_pseudo = retriever.get_top_docs(questions_tensor.numpy(), 1)
        questions = expand_questions(questions, top_ids_and_scores_pseudo, all_passages)
        questions_tensor = retriever.generate_question_vectors(questions)

    if args.out_encoded_question_file:
        print("Generate question embeddings!")
        question_embeddings = []
        for qid, question_tensor in zip(q_ids, questions_tensor):
            question_embeddings.append((qid, question_tensor))

        with open(args.out_encoded_question_file, "wb") as f:
            pickle.dump(question_embeddings, f)
        print(f"Question embeddings saved at {args.out_encoded_question_file}")


    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)

    all_passages = load_passages(args.ctx_file)

    if len(all_passages) == 0:
        raise RuntimeError('No passages data found. Please specify ctx_file param properly.')

    if args.out_file:
        save_results(all_passages, questions, q_ids, top_ids_and_scores, args.out_file, args.out_file_prettified)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--qa_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file', required=True, type=str, default=None,
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--encoded_ctx_file', type=str, default=None,
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--out_file', type=str, default=None, help='output .json file path to write results to')
    parser.add_argument('--out_file_prettified', type=str, default=None, help='same content as out_file but in prettier format')
    parser.add_argument('--n-docs', type=int, default=200, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')
    parser.add_argument("--embed_title", type=bool, default=True, help='Embed title to encoding')
    parser.add_argument("--out_encoded_question_file", type=str, default=None, help="Encoded question or question embedding output filename")
    parser.add_argument("--use_pseudo_feedback", action='store_true', help="Enable pseudo relevance feedback")


    args = parser.parse_args()

    setup_args_gpu(args)
    print_args(args)
    main(args)
