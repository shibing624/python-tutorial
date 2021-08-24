# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import logging

import kashgari
import numpy as np
from kashgari.embeddings import BERTEmbedding

logging.basicConfig(level=logging.DEBUG)


# bert_model_path = os.path.join(utils.get_project_path(), 'tests/test-data/bert')


def cos_dist(emb_1, emb_2):
    """
    calc cos distance
    :param emb_1: numpy.array
    :param emb_2: numpy.array
    :return: cos score
    """
    num = float(np.sum(emb_1 * emb_2))
    denom = np.linalg.norm(emb_1) * np.linalg.norm(emb_2)
    cos = num / denom if denom > 0 else 0.0
    return cos


b = BERTEmbedding(task=kashgari.CLASSIFICATION,
                  model_folder='/Users/xuming06/Codes/bert/data/chinese_L-12_H-768_A-12',
                  sequence_length=12)

# from kashgari.corpus import SMP2018ECDTCorpus

# test_x, test_y = SMP2018ECDTCorpus.load_data('valid')

# b.analyze_corpus(test_x, test_y)
data1 = '湖 北'.split(' ')
data3 = '纽 约'.split(' ')
data2 = '武 汉'.split(' ')
data4 = '武 汉'.split(' ')
data5 = '北 京'.split(' ')
data6 = '武 汉 地 铁'.split(' ')
sents = [data1, data3, data4, data5, data6]
doc_vecs = b.embed(sents, debug=True)

tokens = b.process_x_dataset([['语', '言', '模', '型']])[0]
target_index = [101, 6427, 6241, 3563, 1798, 102]
target_index = target_index + [0] * (12 - len(target_index))
assert list(tokens[0]) == list(target_index)
print(tokens)
print(doc_vecs)
print(doc_vecs.shape)
print(doc_vecs[0])
print(doc_vecs[0][0])

query_vec = b.embed([data2])[0]
query = '武 汉'
# compute normalized dot product as score
for i, sent in enumerate(sents):
    d = b.embed([sent])[0]
    c = cos_dist(d, query_vec)
    print('q:%s, d:%s, s:%s' % (query, sent, c))
