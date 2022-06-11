# DPR
The code is based on [mDPR](https://github.com/AkariAsai/CORA/main/mDPR) and with some modifications for research adjustment.

## Dataset format
Please check retriever input data format section on [this link](https://github.com/facebookresearch/DPR). The required field is only question and positive_ctxs. The rests are optional.

## Training
For the training process you can check train_dense_encoder.py file. There're several parameters that can be adjusted for your experiment needs. As for this experiment here's the command I use.
```
python train_dense_encoder.py --output_dir <where to store model checkpoints for each epoch> --hard_negatives <number of hard negatives for each query that is used for training> --pretrained_model_cfg <model checkpoint file or bert variant name> --sequence_length <sequence length or tokens length maximum> --train_file <json training file> --dev_file <json validation file> --batch_size <training batch size> --dev_batch_size <validation batch size> --seed <random seed, please set this param to ensure reproducibility> --num_train_epochs <number of epoch/iteration for training> --do_lower_case   
```
Here's the complete command for this experiment
```
python train_dense_encoder.py --output_dir ./dpr-checkpoints/ --hard_negatives 0 --pretrained_model_cfg indobenchmark/indobert-base-p2 --sequence_length 512 --train_file ./tydi-indo-train.json --dev_file ./tydi-indo-dev.json \
--batch_size 16 --dev_batch_size 8 --seed 42 --num_train_epochs 10 --do_lower_case
```

## Generate passage embeddings from corpus
After you've finetuned your DPR, the next step is to generate passage embedding using fine-tuned passage encoder on DPR. Here's the command I use for this experiment
```
python generate_dense_embeddings.py --out_file <name-of-passage-embeddings> --ctx_file <corpus-file> --model_file <model checkpoint that you want to use> --do_lower_case --batch_size <same as before> --embed_title <whether to encode title of your passage or not>
```
Here's the complete command for this experiment.
```
python generate_dense_embeddings.py --out_file tydi_passage_embeddings --ctx_file corpus-syifa-normalized.tsv  --model_file ./dpr-checkpoints/dpr_biencoder.8.1226 --do_lower_case --batch_size 128 --embed_title True
```
The corpus should be formatted in tsv with following format
```
id text   title
id text   title
....................
```
## Testing
After you've obtained the passage embedding, the final step is to retrieval result from query on test set. Here's the example command I use for this experiment
```
python dense_retriever.py --out_file <retrieval result> --out_file_prettified <retrieval result but in prettified json format>  --qa_file <test query file in json> --ctx_file <corpus file> --encoded_ctx_file "<name-of-passage-embeddings>" --n-docs <number of documents that'll be etrieved>  --model_file <model checkpoint>  --do_lower_case  --batch_size <batch size for test> --embed_title <whether to embed query title or not> --hnsw_index <variant of index, i use this because it yields best performance> --out_encoded_question_file <question embedding file>
```
Here's the complete example command for this experiment
```
python dense_retriever.py --out_file query_result_tydi.json --out_file_prettified query_result_tydi_normalized_prettified.json --qa_file question-syifa-test.json --ctx_file corpus-syifa-normalized.tsv --encoded_ctx_file "tydi_passage_embeddings" --n-docs 10 --model_file ./dpr-checkpoints/dpr_biencoder.8.1226--do_lower_case --batch_size 8 --embed_title True --hnsw_index --out_encoded_question_file question-embedding-tydi
```
If you want to run DPR-PRF, add parameter --use_pseudo_feedback when you run dense_retriever.py 
Here is the standard format for qa_file
```
[
  {
	"question": {
        "id": ...,
        "title": ...,
        "text": ...
        },
	"ctxs": [{
        "id": "...",
		"title": "...",
		"text": "..."
	}],
    "neg_ctxs": [{
        "id": "...",
		"title": "...",
		"text": "..."
	}]
  },
  ...
]
```

## Hybrid Model
Check out hybrid_dpr_sparse folder