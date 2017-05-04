# -*- coding: utf-8 -*-
"""
@description: 命名实体识别
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import nltk


def ie_preprocess(doc):
    sentences = nltk.sent_tokenize(doc)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]


sentence = [('the', 'DT'), ('little', 'JJ'), ('yellow', 'JJ'),
            ('dog', 'NN'), ('barked', 'VBD'), ('at', 'IN'), ('the', 'DT'), ('cat', 'NN')]
grammar = 'NP:{<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)
result.draw()

sentence = [('another', 'DT'), ('sharp', 'JJ'), ('dive', 'NN'), ('trade', 'NN'),
            ('figures', 'NNS'), ('any', 'DT'), ('new', 'JJ'), ('policy', 'NN'),
            ('measures', 'NNS'), ('earlier', 'JJR'), ('stages', 'NNS'), ('Panamanian', 'JJ'),
            ('dictator', 'NN'), ('Manuel', 'NNP'), ('Noriega', 'NNP')]
grammar = 'NP:{<DT>?<JJ.*>*<NN.*>+}'
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)
# result.draw()

# 在已标注的语料库中提取匹配的特定的词性标记序列的短语
from nltk.corpus import conll2000

print(conll2000.chunked_sents('train.txt')[99])


# 使用unigram标注器对名词短语分块
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
unigram_chunker = UnigramChunker(train_sents)
print('unigram_chunker', unigram_chunker.evaluate(test_sents))


# 二阶
class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


bigram_chunker = BigramChunker(train_sents)
print('bigram_chunker', bigram_chunker.evaluate(test_sents))


# 连续分类器对名词短语分块
def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sentence[i - 1]
    return {'pos': pos, 'word': word, 'prevpos': prevpos}


class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(
            train_set, trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]


chunker = ConsecutiveNPChunker(train_sents)
print('ConsecutiveNPChunker',chunker.evaluate(test_sents))
