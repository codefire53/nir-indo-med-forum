# monoBERT
This directory contains imlementation for monoBERT reranker model.

## Training
To run training for monoBERT, run this command
```
python monobert.py --model_name_or_checkpoint_dir <this can be filled by the checkpoint model directory or the name of the pretrained model stored in huggingface> --max_seq_length <maximum token length> --train_file <training data json file which has the same format as dpr's> --dev_file <development data json file which has the same format as dpr's> --out_dir <directory that we wanna store the model checkpoints for each epoch> --do_train --num_train_epochs <number of training epochs> --train_batch_size <batch size for training data> --dev_batch_size <batch size for development data> --seed <random seed> --plotting_dir <directory name that we wanna use to store the loss plot for throughout the training epochs> --learning_rate <training learning rate>
```

## Inference/Reranking
To run reranking for monoBERT, run this command
```
python monobert.py --model_name_or_checkpoint_dir <this can be filled by the checkpoint model directory or the name of the pretrained model stored in huggingface> --max_seq_length <maximum token length> --test_file <query file which has same format as dpr's that we wanna to rerank> --topk <top k passages for each query> --out_rerank_file <reranking results json filename>
```