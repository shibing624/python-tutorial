# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import fasttext

classifier = fasttext.supervised('train_sample.txt', 'classify_model', label_prefix='__label__')
result = classifier.test('test_sample.txt')
print('P@1:', result.precision)
print('R@1:', result.recall)
print('Number of examples:', result.nexamples)

texts = ['吃 什么 止泻 快 _ 宝宝 拉肚子 _ 酸味 重 _ 专题 解答 ', '增高 _ 正确 长高 方法 _ 刺激 骨骼 二次 生长发育   增高 精准 找到 长高 办法   ,   有助 孩子 长高 的 方法   ,']
labels = classifier.predict(texts)
print(labels)

# Or with the probability
labels = classifier.predict_proba(texts)
print(labels)

labels = classifier.predict(texts, k=3)
print(labels)

# Or with the probability
labels = classifier.predict_proba(texts, k=3)
print(labels)
