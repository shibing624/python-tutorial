# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

from transformers import pipeline
from transformers import AutoModelForQuestionAnswering, BertTokenizer
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
bert_model_dir = os.path.expanduser('~/.pycorrector/datasets/bert_models/chinese_finetuned_lm/')
print(bert_model_dir)
# model = AutoModelForQuestionAnswering.from_pretrained(bert_model_dir)
# tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
nlp = pipeline("question-answering",
               model=bert_model_dir,
               tokenizer=bert_model_dir,
               device=-1,  # gpu device id
               )
context = r"""
大家好，我是张亮，目前任职当当架构部架构师一职，也是高可用架构群的一员。我为大家提供了一份imagenet数据集，希望能够为图像分类任务做点贡献。
"""

# context = ' '.join(list(context))

result = nlp(question="张亮在哪里任职?", context=context)
print(
    f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
result = nlp(question="为图像分类提供了什么数据集?", context=context)
print(
    f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")


# Custom
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
# tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
# model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained(bert_model_dir)
tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

text = r"""
Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""
questions = [
    "How many pretrained models are available in Transformers?",
    "What does Transformers provide?",
    "Transformers provides interoperability between which frameworks?",
]
for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print(f"Question: {question}")
    print(f"Answer: {answer}")