# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

from transformers import pipeline

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
bert_model_dir = os.path.expanduser('~/.pycorrector/datasets/bert_models/chinese_finetuned_lm/')
print(bert_model_dir)
nlp = pipeline("fill-mask",
               model=bert_model_dir,
               tokenizer=bert_model_dir,
               device=-1,  # gpu device id
               )
from pprint import pprint

pprint(nlp(f"张亮在哪里任{nlp.tokenizer.mask_token}?"))
print("*" * 42)
pprint(nlp(f"少先队员应该为老人让{nlp.tokenizer.mask_token}?"))
print("*" * 42)
pprint(nlp(f"明天天{nlp.tokenizer.mask_token}很好?"))
print("*" * 42)
pprint(nlp(f"明天心{nlp.tokenizer.mask_token}很好?"))

# custom
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
# model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
tokenizer = AutoTokenizer.from_pretrained(bert_model_dir)
model = AutoModelWithLMHead.from_pretrained(bert_model_dir)
sequence = f"明天天{nlp.tokenizer.mask_token}很好."
input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
token_logits = model(input).logits
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
