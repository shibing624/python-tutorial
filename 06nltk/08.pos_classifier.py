# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import nltk
from nltk.corpus import brown


# pos features
def pos_features(sentence, i):
    features = {'suffix(1)': sentence[i][-1:],
                'suffix(2)': sentence[i][-2:],
                'suffix(3)': sentence[i][-3:]}
    if i == 0:
        features['prev_word'] = '<START>'
    else:
        features['prev_word'] = sentence[i - 1]
    return features


feature_data = pos_features(brown.sents()[0], 8)
print(feature_data)
tagged_sents = brown.tagged_sents(categories='news')
featuresets = []
for tagged_sent in tagged_sents:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        featuresets.append((pos_features(untagged_sent, i), tag))
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
pos_classifier_nb_rate = nltk.classify.accuracy(classifier, test_set)
print('pos_classifier_nb_rate', pos_classifier_nb_rate)  # 0.789


# optimize classifier
def pos_features(sentence, i, history):
    features = {'suffix(1)': sentence[i][-1:],
                'suffix(2)': sentence[i][-2:],
                'suffix(3)': sentence[i][-3:]}
    if i == 0:
        features['prev-word'] = '<START>'
        features['prev-tag'] = '<START>'
    else:
        features['prev-word'] = sentence[i - 1]
        features['prev-tag'] = history[i - 1]
    return features


class ConsecutivePosTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featuresets = pos_features(untagged_sent, i, history)
                train_set.append((featuresets, tag))
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featuresets = pos_features(sentence, i, history)
            tag = self.classifier.classify(featuresets)
            history.append(tag)
        return zip(sentence, history)


tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.1)
train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
tagger = ConsecutivePosTagger(train_sents)
print('ConsecutivePosTagger', tagger.evaluate(test_sents))
