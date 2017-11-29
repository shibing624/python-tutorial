# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/11/29
# Brief:
import nltk
import time
import pickle
from nltk.corpus import brown

from model.ngram import MLENgramModel
from model.counter import build_vocabulary
from model.counter import count_ngrams

docs = [brown.words(categories='news')]
print("docs success", time.time())
vocab = build_vocabulary(2, *docs)
print("vocab success", time.time())
counter = count_ngrams(3, vocab, *docs)
print(counter)
lm = MLENgramModel(counter)
print(lm)
print("lm success", time.time())
entro = lm.entropy('male female male female')
print(entro)

entro = lm.entropy('male female male male')
print(entro)

perplex = lm.perplexity("My goal do to build a language model.")
print(perplex)

perplex = lm.perplexity("My goal do to build an language model.")
print(perplex)
print("perplexity success", time.time())

output = open('ngram_model.pkl', 'wb')
pickle.dump(lm, output)
output.close()
