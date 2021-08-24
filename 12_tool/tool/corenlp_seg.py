# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

# from nltk.parse.corenlp import CoreNLPParser
# stanford = CoreNLPParser()
# str = 'proved to be fake, made-up'
# token = list(stanford.tokenize(str))
# print(token)


# from nltk.tokenize import word_tokenize
# token = word_tokenize(str)
# print(token)

from collections import Counter
a = ['w','good','good','w','ww']
b = Counter(a)
print(b)
b.update(a)
print(b)
print(b.most_common())