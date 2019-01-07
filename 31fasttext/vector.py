# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import fasttext

# Skipgram model
model = fasttext.skipgram('train_sample.txt', 'model')
print(len(model.words))  # list of words in dictionary
print(model.words)
print(model['及'])
del model

model = fasttext.load_model('model.bin')
print(model['及'])  # get the vector of the word 'king'
