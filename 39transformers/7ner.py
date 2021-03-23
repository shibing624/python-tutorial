# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
model_dir = os.path.expanduser('/Users/xuming06/Documents/Data/chinese-xlnet-base/')
print(model_dir)

nlp = pipeline("ner",
               model=model_dir, tokenizer=model_dir)
sequence = "王宏伟来自北京，是个警察，喜欢去王府井游玩儿。"
print(nlp(sequence))
# custom

# model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
label_list = [
    "O",  # Outside of a named entity
    "B-PER",  # Beginning of a person's name right after another person's name
    "I-PER",  # Person's name
    "B-ORG",  # Beginning of an organisation right after another organisation
    "I-ORG",  # Organisation
    "B-LOC",  # Beginning of a location right after another location
    "I-LOC"  # Location
]
sequence = "王宏伟来自北京，是个警察，喜欢去王府井游玩儿。"
# Bit of a hack to get the tokens with the special tokens
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="pt")
outputs = model(inputs).logits
predictions = torch.argmax(outputs, dim=2)
print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])
