# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import nltk
from nltk.corpus import names


def gender_features(word):
    return {'last_letter': word[-1]}


print(gender_features('sherk'))

names = ([(name, 'male') for name in names.words('male.txt')] +
         [(name, 'female') for name in names.words('female.txt')])
featuresets = [(gender_features(n), g) for (n, g) in names]
train_set_orginal, test_set_orginal = featuresets[500:], featuresets[:500]
from nltk.classify import apply_features

train_set = apply_features(gender_features, names[500:])
test_set = apply_features(gender_features, names[:500])

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(classifier.classify(gender_features('Neo')))
print(classifier.classify(gender_features('Trinity')))
print(nltk.classify.accuracy(classifier, test_set))

# check
classifier.show_most_informative_features(5)


def gender_features2(name):
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    return features


print(gender_features2('John'))
# use gender_features2 test NB
train_set = apply_features(gender_features2, names[500:])
test_set = apply_features(gender_features2, names[:500])
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

print(len(names))
train_names = names[1500:]
devtest_names = names[500:1500]
test_names = names[:500]
train_set = [(gender_features(n), g) for (n, g) in train_names]
devtest_set = [(gender_features(n), g) for (n, g) in devtest_names]
test_set = [(gender_features(n), g) for (n, g) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))
# error list
errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append((tag, guess, name))
for (tag, guess, name) in sorted(errors):
    print('correct=%-8s guess=%-8s name=%-30s' % (tag, guess, name))


def gender_suffix_features(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:]}


train_set = [(gender_suffix_features(n), g) for (n, g) in train_names]
devtest_set = [(gender_suffix_features(n), g) for (n, g) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('gender_suffix_features',nltk.classify.accuracy(classifier, devtest_set))
