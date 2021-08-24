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
nlp = pipeline("sentiment-analysis",
               model=bert_model_dir,
               tokenizer=bert_model_dir,
               device=-1,  # gpu device id
               )

result = nlp("我爱你")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

result = nlp("我恨你")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# Custom
from transformers import AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(bert_model_dir)
print("token ok")
model = AutoModelForSequenceClassification.from_pretrained(bert_model_dir)
print("model ok")

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")
classes = ["not paraphrase", "is paraphrase"]
sequence_0 = "中国首都是北京"
sequence_1 = "苹果有益于你的身体健康"
sequence_2 = "北京是在北回归线以南的城市"
paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")
paraphrase_classification_logits = model(**paraphrase).logits
not_paraphrase_classification_logits = model(**not_paraphrase).logits
paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]
# Should be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
# Should not be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
