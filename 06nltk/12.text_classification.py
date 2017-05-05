# -*- coding: utf-8 -*-
"""
@description: 文本分类
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

from nltk.corpus import stopwords

def bag_of_words(words):
    return dict([(word, True) for word in words])


print(bag_of_words(['the', 'quick', 'brown', 'fox']))


def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))


def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)


set_out = bag_of_non_stopwords(['the', 'quick', 'brown', 'fox'])
print(set_out)
# {'quick': True, 'brown': True, 'fox': True}
# 'the' is in the stopwords

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)


bi_out = bag_of_bigrams_words(['the', 'quick', 'brown', 'fox'])
print(bi_out)
# {'the': True, 'quick': True, 'brown': True, 'fox': True, ('brown', 'fox'): True, ('quick', 'brown'): True, ('the', 'quick'): True}


# NB
import collections


def label_feats_from_corpus(corp, feature_detector=bag_of_words):
    label_feats = collections.defaultdict(list)
    for label in corp.categories():
        for fileid in corp.fileids(categories=[label]):
            feats = feature_detector(corp.words(fileids=[fileid]))
            label_feats[label].append(feats)
        return label_feats


def split_label_feats(lfeats, split=0.75):
    train_feats = []
    test_feats = []
    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
    return train_feats, test_feats


from nltk.corpus import movie_reviews

cate = movie_reviews.categories()
print(cate)
lfeats = label_feats_from_corpus(movie_reviews)
keys = lfeats.keys()
print(keys)
train_feats, test_feats = split_label_feats(lfeats, split=0.75)
print(len(train_feats))
print(len(test_feats))

from nltk.classify import NaiveBayesClassifier

nb_classifier = NaiveBayesClassifier.train(train_feats)
print(nb_classifier.labels())

negfeat = bag_of_words(['the', 'plot', 'was', 'ludicrous'])
out = nb_classifier.classify(negfeat)
print(out)

posfeat = bag_of_words(['kate', 'winslet', 'is', 'accessible'])
print(nb_classifier.classify(posfeat))

from nltk.classify.util import accuracy

print(accuracy(nb_classifier, test_feats))
probs = nb_classifier.prob_classify(test_feats[0][0])
print(probs.samples())
print(probs.max())
print(probs.prob('pos'))
print(probs.prob('neg'))

