# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os
import pickle

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_filepath = 'tfidf.pkl'
file_path = 'yl_10.txt'
titles = []
with open(file_path, 'r', encoding='utf-8') as f:
    count = 0
    for line in f:
        line = line.strip()
        cols = line.split('\t')
        userid = cols[0]
        titles.append(userid)
        content = " ".join(cols[1:])
        print(count, userid, content)
        count += 1


def read_words(file_path):
    words = set()
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            words.add(line)
    return words


def trim_stopwords(words, stop_words_set):
    """
    去除切词文本中的停用词
    :param words:
    :param stop_words_set:
    :return:
    """
    new_words = []
    for w in words:
        if w in stop_words_set:
            continue
        new_words.append(w)
    return new_words


def feature():
    docs = []
    word_set = set()
    stopwords = read_words("../../data/stopword.txt")
    with open(file_path, 'r', encoding='utf-8')as f:
        for line in f:
            line = line.strip()
            cols = line.split("\t")
            userid = cols[0]
            content = " ".join(cols[1:])
            content = content.lower()
            words = jieba.lcut(content)
            doc = trim_stopwords(words, stopwords)
            docs.append(" ".join(doc))
            word_set |= set(doc)
    print('word set size:%s' % len(word_set))
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.1, analyzer='word', ngram_range=(1, 2),
                                       vocabulary=list(word_set))
    return tfidf_vectorizer, docs


if not os.path.exists(tfidf_filepath):
    tfidf_vectorizer, docs = feature()
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)  # fit the vectorizer to synopses
    # terms is just a 集合 of the features used in the tf-idf matrix. This is a vocabulary
    terms = tfidf_vectorizer.get_feature_names()  # 长度258
    print(terms)
    print('feature name size:%s' % len(terms))

    with open(tfidf_filepath, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
else:
    with open(tfidf_filepath, "rb") as f:
        tfidf_matrix = pickle.load(f)

print(tfidf_matrix.shape)  # (10, 258)：10篇文档，258个feature

from sklearn.metrics.pairwise import cosine_similarity

# Note that 有了 dist 就可以测量任意两个或多个概要之间的相似性.
# cosine_similarity返回An array with shape (n_samples_X, n_samples_Y)
dist = 1 - cosine_similarity(tfidf_matrix)

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

zh_font = FontProperties(fname='/Library/Fonts/Songti.ttc')
# Perform Ward's linkage on a condensed distance matrix.
# linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
# Method 'ward' requires the distance metric to be Euclidean
linkage_matrix = linkage(dist, method='ward', metric='euclidean', optimal_ordering=False)
# linkage_matrix_not = linkage(dist, method='ward', metric='euclidean', optimal_ordering=False)
# Z[i] will tell us which clusters were merged, let's take a look at the first two points that were merged
# We can see that ach row of the resulting array has the format [idx1, idx2, dist, sample_count]
print(linkage_matrix)


def show_link():
    plt.figure(figsize=(25, 10))
    plt.title('中文文本层次聚类树状图', fontproperties=zh_font)
    plt.xlabel('微博标题', fontproperties=zh_font)
    plt.ylabel('距离（越低表示文本越类似）', fontproperties=zh_font)
    dendrogram(
        linkage_matrix,
        labels=titles,
        leaf_rotation=-70,  # rotates the x axis labels
        leaf_font_size=12  # font size for the x axis labels
    )
    plt.show()
    plt.close()


show_link()

from sklearn.cluster import MiniBatchKMeans

n_clusters = 3
X = tfidf_matrix
kmeans = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=100,
                         n_init=10, max_no_improvement=10, verbose=0)
kmeans.fit(X)
print(kmeans.cluster_centers_)
labels = kmeans.labels_
print(labels)
from collections import Counter

# good_columns = X._get_numeric_data().dropna(axis=1)

print(Counter(kmeans.labels_))
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD()
plot_columns = svd.fit_transform(X)
plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=labels)
plt.show()
