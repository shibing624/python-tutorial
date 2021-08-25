# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""



import logging

import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence']]
# train model
model = gensim.models.Word2Vec(sentences, min_count=1)
print(model['first'])
print(model.similarity('first', 'second'))
