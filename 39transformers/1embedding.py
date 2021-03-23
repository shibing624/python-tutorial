# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
model_dir = os.path.expanduser('~/.pycorrector/datasets/bert_models/chinese_finetuned_lm/')
print(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
print("token ok")
model = AutoModel.from_pretrained(model_dir)
print("model ok")
# tensor([[ 101, 3217, 4697,  679, 6230, 3236,  102]])
inputs = tokenizer('春眠不觉晓', return_tensors='pt')
outputs = model(**inputs)  # shape (1, 7, 768)
print(outputs)
v = torch.mean(outputs[0], dim=1)  # shape (1, 768)
# print(v)


def sentence_embedding(sentence):
    input_ids = tokenizer(sentence, return_tensors='pt')
    o = model(**input_ids)
    return torch.mean(o[0], dim=1)


sentences = ['春眠不觉晓', '大梦谁先觉', '浓睡不消残酒', '东临碣石以观沧海']

with torch.no_grad():
    vs = [sentence_embedding(sentence).numpy() for sentence in sentences]
    nvs = [v / np.linalg.norm(v) for v in vs]  # normalize each vector
    m = np.array(nvs).squeeze(1)  # shape (4, 768)
    print(np.around(m @ m.T, decimals=2))  # pairwise cosine similarity
