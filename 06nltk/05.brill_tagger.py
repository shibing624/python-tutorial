# -*- coding: utf-8 -*-
"""
@description: base on rule tagger: Brill tagger 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理
import nltk
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

size = int(len(brown_tagged_sents) * 0.9)
print(size)  # 4160

train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

text = """the board's action shows it is up againest in 
our complex maze of regulatory laws."""
tokens = text.split()

# To train and test using Brown Corpus.
from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word
from nltk.tag.brill import BrillTagger
unigram_tagger = nltk.UnigramTagger(train_sents)
Template._cleartemplates() #clear any templates created in earlier tests
templates = [Template(Pos([-1])), Template(Pos([-1]), Word([0]))]
brill = BrillTagger(unigram_tagger,templates)
print(brill.tag(tokens))

