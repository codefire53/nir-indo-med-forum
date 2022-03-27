"""Code to train and eval a BERT passage re-ranker on the MS MARCO dataset."""



import os
import time
import random

import numpy as np

from operator import itemgetter

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AdamW, AutoConfig, get_linear_schedule_with_warmup, AutoTokenizer

class MonoDataset(Dataset):
    def __init__(self, tokenizer, input_json_path, max_seq_length):
        self.max_seq_length = max_seq_length
        
        self.input_ids = []
        self.attn_masks = []
        self.input_segments = []
        self.labels = []
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        for qrow in tqdm(data):
            q_full_text = qrow['question']
            q_encoded = tokenizer.encode_plus(q_full_text, add_special_tokens = True, max_length = max_seq_length//4, pad_to_max_length=True, return_attention_mask=True)
            for prow in qrow['positive_ctxs']:
                p_title = prow['title']
                p_text = prow['text']
                full_text = f"{p_title}. {p_text}"
                p_encoded = tokenizer.encode_plus(full_text, add_special_tokens = True, max_length = max_seq_length*3//4, pad_to_max_length=True, return_attention_mask=True)
                #print(q_encoded['input_ids']+p_encoded['input_ids'][1:])
                self.input_ids.append(torch.tensor(q_encoded['input_ids']+p_encoded['input_ids'][1:], dtype=torch.int32))
                self.attn_masks.append(torch.tensor(q_encoded['attention_mask']+p_encoded['attention_mask'][1:], dtype=torch.int32))
                self.input_segments.append(torch.cat((torch.zeros_like(torch.tensor(q_encoded['input_ids'], dtype=torch.int32)), torch.ones_like(torch.tensor(p_encoded['input_ids'][1:], dtype=torch.int32))), -1))
                #print(self.input_segments)
                self.labels.append(1)
            for prow in qrow['negative_ctxs']:
                p_title = prow['title']
                p_text = prow['text']
                full_text = f"{p_title}. {p_text}"
                p_encoded = tokenizer.encode_plus(full_text, add_special_tokens = True, max_length = max_seq_length*3//4, pad_to_max_length=True, return_attention_mask=True)
                self.input_ids.append(torch.tensor(q_encoded['input_ids']+p_encoded['input_ids'][1:], dtype=torch.int32))
                self.attn_masks.append(torch.tensor(q_encoded['attention_mask']+p_encoded['attention_mask'][1:], dtype=torch.int32))
                self.input_segments.append(torch.cat((torch.zeros_like(torch.tensor(q_encoded['input_ids'], dtype=torch.int32)), torch.ones_like(torch.tensor(p_encoded['input_ids'][1:], dtype=torch.int32))), -1))
                self.labels.append(0)
        indexes = list(range(len(self.input_ids))) 
        random.shuffle(indexes)
        self.input_ids = [self.input_ids[idx] for idx in indexes]
        self.attn_masks = [self.attn_masks[idx] for idx in indexes]
        self.input_segments = [self.input_segments[idx] for idx in indexes]
        self.labels = [self.labels[idx] for idx in indexes]
        
        self.input_ids = torch.stack(self.input_ids)
        self.attn_masks =  torch.stack(self.attn_masks)
        self.input_segments =  torch.stack(self.input_segments)
        self.labels = torch.tensor(self.labels)
        
    def __len__(self):  
        return len(self.input_ids)

    def __getitem__(self, item):
    
        ret_val = {
            "input_ids": self.input_ids[item],
            "attention_mask": self.attn_masks[item],
            "input_segments": self.input_segments[item],
            "labels": self.labels[item]
        }

        return ret_val

def visualize_loss(save_dir, train_loss, val_loss, epoch):
    n_epoch  = list(range(epoch))
    plt.plot(n_epoch, train_loss, label='training loss')
    plt.plot(n_epoch, val_loss, label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir,'epoch.png'))

def train(args):
    model  = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_checkpoint_dir, num_labels=2, output_attentions=False, output_hidden_states=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_checkpoint_dir, do_lower_case=True)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    train_dataset = MonoDataset(tokenizer, args.train_file, args.max_seq_length)
    dev_dataset  = MonoDataset(tokenizer, args.dev_file, args.max_seq_length)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)
    dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=args.dev_batch_size)

    total_steps = len(train_dataloader)*args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    
    epochs = args.num_train_epochs
    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        total_train_loss = 0.0
        model.train()
        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].to(args.device)
            attention_masks = batch["attention_mask"].to(args.device)
            token_type_ids = batch["input_segments"].to(args.device)
            labels = batch["labels"].to(args.device)
            model.zero_grad()

            output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks, labels=labels)
            loss = output.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss/len(train_dataloader)
        train_loss_per_epoch.append(avg_train_loss)
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        
        epoch_output = os.path.join(args.out_dir, f"checkpoint-{epoch}")
        if not os.path.exists(epoch_output):
            os.makedirs(epoch_output)
        print(f"Saving model to {epoch_output}")
        
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(epoch_output)
        tokenizer.save_pretrained(epoch_output)
        torch.save(args, os.path.join(epoch_output, 'training_args.bin'))
        
        print("")
        print("Validation...")
        model.eval()
        total_val_loss = 0.0
        for batch in tqdm(dev_dataloader):
            input_ids = batch["input_ids"].to(args.device)
            attention_masks = batch["attention_mask"].to(args.device)
            token_type_ids = batch["input_segments"].to(args.device)
            labels = batch["labels"].to(args.device)
            model.zero_grad()

            output  = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks, labels=labels)
            loss = output.loss
            total_val_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_val_loss = total_val_loss/len(dev_dataloader)
        val_loss_per_epoch.append(avg_val_loss)
        print("Average validation loss: {0:.2f}".format(avg_val_loss))
        if args.plotting_dir:
            visualize_loss(args.plotting_dir, train_loss_per_epoch, val_loss_per_epoch, epoch+1)
    print("")
    print("Training finished!")

def rerank(args):
    model  = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_checkpoint_dir, num_labels=2, output_attentions=False, output_hidden_states=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_checkpoint_dir, do_lower_case=True)
    model.to(args.device)
    with open(args.test_file, 'r') as f:
        data = json.load(f)
    model.eval()
    for qidx, qrow in tqdm(enumerate(data)):
        q_text = f"{qrow['question']['title']}. {qrow['question']['text']}"
        q_encoded = tokenizer.encode_plus(q_text, add_special_tokens = True, max_length = args.max_seq_length//4, pad_to_max_length=True, return_attention_mask=True)
        for pidx,prow in enumerate(qrow['ctxs']):
            p_text =  f"{prow['title']}. {prow['text']}"
            p_encoded = tokenizer.encode_plus(p_text, add_special_tokens = True, max_length = args.max_seq_length*3//4, pad_to_max_length=True, return_attention_mask=True)
            input_ids = torch.tensor(q_encoded['input_ids']+p_encoded['input_ids'][1:], dtype=torch.int32).unsqueeze(0).to(args.device)
            attn_masks = torch.tensor(q_encoded['attention_mask']+p_encoded['attention_mask'][1:], dtype=torch.int32).unsqueeze(0).to(args.device)
            input_segments = torch.cat((torch.zeros_like(torch.tensor(q_encoded['input_ids'], dtype=torch.int32)), torch.ones_like(torch.tensor(p_encoded['input_ids'][1:], dtype=torch.int32))), -1).unsqueeze(0).to(args.device)
            output = model(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=input_segments)
            prob = output.logits
            if prob.size(1) > 1:
                score = torch.nn.functional.log_softmax(prob, 1)[0, -1].item() 
            else:
                score = prob.item()
            data[qidx]['ctxs'][pidx]['rerank_score'] = score
        data[qidx]['ctxs'] = sorted(data[qidx]['ctxs'], key=itemgetter('rerank_score'), reverse=True)[:args.topk]
    with open(args.out_rerank_file, 'w') as f:
        json.dump(data, f)
    print(f"Rerank result is written to {args.out_rerank_file}")

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_checkpoint_dir", default = 'indobenchmark/indobert-base-p2',type = str)
    parser.add_argument("--max_seq_length", default = 512, type = int)
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--dev_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--dev_batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--plotting_dir", type=str, default=None)
    parser.add_argument("--out_rerank_file", type=str, default="./monobert-rerank-results.json")

    args = parser.parse_args()
    args.device = device
    args.use_cuda = torch.cuda.is_available()


    #init seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.do_train:
        assert args.train_file is not None and args.dev_file is not None and args.out_dir is not None
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        train(args)

    elif args.test_file:
        rerank(args)

if __name__ == '__main__':
    main()
