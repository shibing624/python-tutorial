# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""


import jieba

text = "哈尔滨工业大学迎来100年华诞，周华主持了会议。乐视赞助了会议"
print(jieba.lcut(text))

import jieba.posseg
print(jieba.posseg.lcut(text))

from fastHan import FastHan

model = FastHan()
sentence = "郭靖是金庸笔下的一名男主。"
answer = model(sentence, target="NER")
print(answer)

with open("dynamic_samples.txt", 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip('\n')
        if line.startswith('#'):
            continue
        parts = line.split('\t')
        idea = parts[0]
        brand = parts[1] if len(parts) == 2 else ''

        words = jieba.posseg.lcut(idea)
        ners = model(idea, target='NER')
        if brand:
            print(brand, ners, words)