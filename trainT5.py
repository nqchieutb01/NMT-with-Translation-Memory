import transformers
from transformers import (
    MT5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, MT5Tokenizer, MT5Config
)

import datasets
import pandas as pd
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
from datasets import load_metric
import gc
import datasets
import os
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"
os.environ["WANDB_DISABLED"] = "true"
!export CUDA_VISIBLE_DEVICES=7
device, use_gpu = ("cuda:7", True) if torch.cuda.is_available() else ("cpu", False)
import json
checkpoint = "/workspace/home/chieunq/viT5-large-intend/checkpoint-12500"
model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
print('load model done')
tokenizer = MT5Tokenizer.from_pretrained(checkpoint)
print('load tokenizer done')

import re
def format(tmp):
    tmp = re.sub('=', 'E', tmp)
    tmp = re.sub('<', 'S', tmp)
    tmp = re.sub('>', 'B', tmp)
    return tmp
def load_data():
    data = []
    with open("data/augument_gold.jsonl",encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if len(line) < 1:
                break
            line = json.loads(line)
            
            data.append(
                {
                    'question': line['input'],
                    'sql': format(line['output'])
                }
            )
    
    print(f'total size of data is {len(data)}')
    
    tdata = pd.DataFrame(data)
    tdata = tdata.reset_index()
    dataset = datasets.Dataset.from_pandas(tdata)

    # don't care about test_size. 
    train = dataset.train_test_split(
        train_size=241, test_size=1, seed=42
    )
    return train
data = load_data()

train_data = data['train']
test_data = data['test']

def format_dataset(example):
     return {'input': example['question'], 'target': example['sql']}
train_data = train_data.map(format_dataset, remove_columns=train_data.column_names)
test_data = test_data.map(format_dataset, remove_columns=test_data.column_names)

def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input'], pad_to_max_length=True, max_length=128)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target'], pad_to_max_length=True, max_length=128)
    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings
train_data = train_data.map(convert_to_features, batched=True, remove_columns=train_data.column_names)
test_data = test_data.map(convert_to_features, batched=True, remove_columns=test_data.column_names)

# columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']

# train_data.set_format(type='torch', columns=columns)
# test_data.set_format(type='torch', columns=columns)
from datasets import load_metric
rouge = load_metric("rouge1.py")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }
data_collator = DataCollatorForSeq2Seq(tokenizer,model=model)
training_args = Seq2SeqTrainingArguments(
    output_dir="viT5-large-intend-continue-1",
    per_device_train_batch_size=1,
    num_train_epochs=4,
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    evaluation_strategy="steps",
    do_train=True,
    do_eval=True,
    logging_steps=241,
    save_strategy="steps",
    save_steps=241,
    eval_steps=241,
    overwrite_output_dir=True,
    save_total_limit=5,
    load_best_model_at_end=True,
    report_to=None,
    #fp16=True, 
)
trainer = Seq2SeqTrainer(
    model=model,
    data_collator = data_collator,
    tokenizer = tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data,
)
trainer.train()
