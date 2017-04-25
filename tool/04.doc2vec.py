# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import gensim
from gensim.models import Doc2Vec

# 获取训练与测试数据及其类别标注
neg_file = 'douban_imdb_data/neg.txt'
pos_file = 'douban_imdb_data/aclImdb/train/pos'
unsup_file = 'douban_imdb_data/aclImdb/train/unsup'
sentences = gensim.models.doc2vec.TaggedLineDocument(neg_file)
model = gensim.models.doc2vec.Doc2Vec(sentences)
model.save('neg.d2v.model')
model = Doc2Vec.load('neg.d2v.model')
sims = model.docvecs.most_similar(9)
print(sims)

print(model.doesnt_match("annoying is this new IMDB rule of requiring".split()))
print(model.doesnt_match(" over was the fact that ".split()))
print(model.doesnt_match("my god this really".split()))
print(model.doesnt_match("I'm sure I missed some plot points".split()))


# print(model.most_similar(positive=['but', 'what'], negative=['fact']))
# print(model.most_similar(positive=['blue', 'shirt'], negative=['blue']))
# print(model.most_similar(positive=['calvin', 'klein'], negative=['tommy']))
# print(model.most_similar(positive=['cotton', 'material'], negative=['polyester']))
# print(model.most_similar(positive=['nike', 'run'], negative=['express']))
#
#
#
# print(model.most_similar_cosmul(positive=['calvin', 'klein'], negative=['tommy']) )
# print(model.most_similar_cosmul(positive=['skinny', 'jean'], negative=['large']) )
# print(model.most_similar_cosmul(positive=['black', 'dress'], negative=['navy']) )
# print(model.most_similar_cosmul(positive=['blue', 'coat'], negative=['yellow']) )
