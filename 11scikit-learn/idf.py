# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["I come to China to travel",
          "This is a car polupar in China",
          "I love tea and Apple ",
          "The work is to write some papers in science"]

vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(corpus)
print(tfidf)
print('vocab:')
print(vectorizer.vocabulary_)
word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
    k_v = dict()
    for j in range(len(word)):
        print(word[j], weight[i][j])
        k_v[word[j]] = weight[i][j]
    sorts = sorted(k_v.items(), key=lambda d:d[1],reverse=True)
    print(sorts[:5])
