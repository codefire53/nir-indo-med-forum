import lucene
import argparse
from org.apache.lucene.analysis.standard import StandardAnalyzer

from org.apache.lucene.document import Document, Field, StringField, TextField

from org.apache.lucene.index import DirectoryReader, IndexReader, IndexWriter, IndexWriterConfig

from org.apache.lucene.queryparser.classic import ParseException, QueryParser

from org.apache.lucene.search import IndexSearcher, Query, ScoreDoc, TopScoreDocCollector
from org.apache.lucene.search.similarities import BM25Similarity, ClassicSimilarity, LMDirichletSimilarity, Similarity

from org.apache.lucene.store import Directory, ByteBuffersDirectory

from typing import List, Tuple

import csv
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def setup_indexer_classes():
	analyzer = StandardAnalyzer()
	index = ByteBuffersDirectory()
	index_writer_config = IndexWriterConfig(analyzer)
	index_writer_config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
	index_writer = IndexWriter(index, index_writer_config)
	return analyzer, index, index_writer_config, index_writer

def create_index(corpus_path, index_writer):
    print("Create Index!")
    passage_cnt = 0
    with open(corpus_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader):
            pid = row[0]
            text = row[1].replace(",", " ")
            title = row[2].replace(",", " ")
            full_text = f"{title}. {text}"
            document = Document()
            document.add(StringField("question_id", pid, Field.Store.YES))
            document.add(TextField("question_title", title, Field.Store.YES))
            document.add(TextField("question_detail", text, Field.Store.YES))
            document.add(TextField("question_title_detail", full_text, Field.Store.YES))
            index_writer.addDocument(document)
            passage_cnt += 1
    print(f"{passage_cnt} passages successfully indexed!")
    index_writer.close()


def question_queries(query_type: str, query_path: str) -> Tuple[List[str], List[dict]] :
	print("Parse queries!")
	query_cnt = 0
	queries = []
	with open(query_path, 'r') as f:
		data = json.load(f)
	for row in tqdm(data):
		title = row['question']['title'].replace("?", "").replace(",", " ")
		text = row['question']['text'].replace("?", "").replace(",", " ")
		query = f"{title}. {text}" if query_type=='titledetail' else title
		queries.append(query)
		query_cnt +=1
	print(f"Total queries: {query_cnt}")
	return queries, data

def setup_searcher_classes_with_similarity(similarity, index_directory, topk):
	index_reader = DirectoryReader.open(index_directory)
	index_searcher = IndexSearcher(index_reader)
	collector = TopScoreDocCollector.create(topk, 100)
	index_searcher.setSimilarity(similarity)
	return index_searcher, collector, index_reader

def get_result(searcher, hits)-> list:
	print("Parse query result")
	results = []
	for hit in hits:
		score = hit.score
		doc_id = hit.doc
		selected_doc = searcher.doc(doc_id)
		doc_id = selected_doc.get("question_id")
		title = selected_doc.get("question_title")
		text = selected_doc.get("question_detail")
		results.append({
			'id': str(doc_id),
			'title': title,
			'text': text
		})
	print("Query has succesfully parseed")
	return results
	

def search(searcher, query, index_directory, topk: int) -> list:        	
	index_searcher, collector, index_reader = setup_searcher_classes_with_similarity(searcher, index_directory, topk)
	index_searcher.search(query, collector)
	hits = collector.topDocs().scoreDocs
	results = get_result(index_searcher, hits)
	index_reader.close()
	return results

def main():
	lucene.initVM(vmargs=['-Djava.awt.headless=true'])
	parser = argparse.ArgumentParser()
	parser.add_argument("--query_path", type=str, required=True)
	parser.add_argument("--corpus_path", type=str, required=True)
	parser.add_argument("--topk", type=int, default=10)
	args = parser.parse_args()
	assert args.query_path.endswith(".json") and args.corpus_path.endswith(".tsv")
	
	experiment_combinations = ['title_title', 'title_titledetail', 'titledetail_titledetail']
	analyzer, index_directory, index_writer_config, index_writer = setup_indexer_classes()
	create_index(args.corpus_path, index_writer)
	searchers = [ClassicSimilarity(), BM25Similarity(), LMDirichletSimilarity()]
	searchers_name = ['ClassicSimilarity', 'BM25Similarity', 'LMDirichletSimilarity'] 
	
	for experiment_combination in tqdm(experiment_combinations):
		print(f"Experiment combination: {experiment_combination}")
		types = experiment_combination.split("_")
		query_type = types[0]
		doc_type = types[1]
		queries, queries_map = question_queries(query_type,  args.query_path)
		search_results = dict()
		for name in searchers_name:
			search_results[name] =  []
		for idx, query in tqdm(enumerate(queries)):
			query_parser = QueryParser("question_title", analyzer) if doc_type.lower() == 'title' else QueryParser("question_title_detail", analyzer)
			query = query_parser.parse(query)
			for sidx, searcher in enumerate(searchers):
				similar_questions = search(searcher, query, index_directory, args.topk)
				search_results[searchers_name[sidx]].append({
					'question': queries_map[idx]['question'],
					'ctxs': similar_questions
				})
		for k, v in search_results.items():
			result_file = f"{k}_{experiment_combination}_top{args.topk}_results.json"
			with open(result_file, 'w') as f:
				json.dump(v, f)
			print(f"Result for {k} scoring for {experiment_combination} experiment is saved at {result_file}")
		

if __name__ == '__main__':
    main()
