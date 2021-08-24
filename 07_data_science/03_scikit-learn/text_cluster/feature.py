# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import math

import jieba
import jieba.analyse
import numpy as np


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


def tfidf(term, doc, word_dict, doc_set):
    tf = float(doc.count(term)) / (len(doc) + 0.001)
    idf = math.log(float(len(doc_set)) / word_dict[term])
    return tf * idf


def idf(term, word_dict, docset):
    idf = math.log(float(len(docset)) / word_dict[term])
    return idf


def get_all_vector(file_path, stop_words):
    names = []
    docs = []
    word_set = set()
    with open(file_path, 'r', encoding='utf-8')as f:
        for line in f:
            line = line.strip()
            cols = line.split("\t")
            userid = cols[0]
            names.append(userid)
            content = " ".join(cols[1:])
            content = content.lower().replace("{地域}{投放地域}", "").replace("{关键词}", "")
            words = jieba.lcut(content)
            doc = trim_stopwords(words, stop_words)
            docs.append(doc)
            word_set |= set(doc)

    docs_vsm = []
    for doc in docs:
        temp_vector = []
        for word in word_set:
            temp_vector.append(doc.count(word) * 1.0)
        # print temp_vector[-30:-1]
        docs_vsm.append(temp_vector)

    docs_matrix = np.array(docs_vsm)
    # print docs_matrix.shape
    # print len(np.nonzero(docs_matrix[:,3])[0])
    column_sum = [float(len(np.nonzero(docs_matrix[:, i])[0])) for i in range(docs_matrix.shape[1])]
    column_sum = np.array(column_sum)
    column_sum = docs_matrix.shape[0] / column_sum
    idf = np.log(column_sum)
    idf = np.diag(idf)
    # print idf.shape
    # row_sum    = [ docs_matrix[i].sum() for i in range(docs_matrix.shape[0]) ]
    # print idf
    # print column_sum
    tfidf = np.dot(docs_matrix, idf)

    return names, tfidf


def gen_sim(A, B):
    num = float(np.dot(A, B.T))
    denum = np.linalg.norm(A) * np.linalg.norm(B)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn
    return sim


def rand_center(data_set, k):
    n = np.shape(data_set)[1]
    centroids = np.mat(np.zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(data_set[:, j])
        rangeJ = float(max(data_set[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


def kmeans(data_set, k):
    m = np.shape(data_set)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = rand_center(data_set, k)
    counter = 0
    while counter <= 50:
        counter += 1
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = gen_sim(centroids[j, :], data_set[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print centroids
        for cent in range(k):  # recalculate centroids
            ptsInClust = data_set[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = np.mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment


if __name__ == "__main__":
    stop_words = read_words("../../data/stopword.txt")
    names, tfidf_mat = get_all_vector("./yl_10.txt", stop_words)
    myCentroids, clustAssing = kmeans(tfidf_mat, 8)
    for label, name in zip(clustAssing[:, 0], names):
        print(label, name)
