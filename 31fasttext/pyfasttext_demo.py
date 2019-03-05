# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: pyfasttext 可以解决fasttext "C++扩展无法分配足够的内存"的问题。总结：1. 训练使用原版fasttext库；2.预测使用pyfasttext
"""

from pyfasttext import FastText
model = FastText('classify_model.bin')
print(model.labels)
print(model.nlabels)
texts = ['吃 什么 止泻 快 _ 宝宝 拉肚子 _ 酸味 重 _ 专题 解答 ', '增高 _ 正确 长高 方法 _ 刺激 骨骼 二次 生长发育   增高 精准 找到 长高 办法   ,   有助 孩子 长高 的 方法   ,']

# Or with the probability
labels = model.predict_proba(texts, k=2)
print(labels)

print(model.predict(texts, k=1))