# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from fastHan import FastHan

model = FastHan()
sentence = "郭靖是金庸笔下的一名男主。"
answer = model(sentence, target="Parsing")
print(answer)
answer = model(sentence, target="NER")
print(answer)

sentence = "一个苹果。"
print(model(sentence, 'CWS'))
model.set_cws_style('cnc')
print(model(sentence, 'CWS'))

sentence = ["我爱踢足球。", "林丹是冠军"]
answer = model(sentence, 'Parsing')
for i, sentence in enumerate(answer):
    print(i)
    for token in sentence:
        print(token, token.pos, token.head, token.head_label)
