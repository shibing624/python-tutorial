# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from transformers import pipeline

translator = pipeline("translation_en_to_de")
print(translator("Hugging Face is a technology company based in New York and Paris", max_length=40))

from transformers import AutoModelWithLMHead, AutoTokenizer

model = AutoModelWithLMHead.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
inputs = tokenizer.encode(
    "translate English to German: Hugging Face is a technology company based in New York and Paris",
    return_tensors="pt")
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)

print(tokenizer.decode(outputs[0]))
