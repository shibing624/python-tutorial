# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import os

import gensim
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from sklearn.cross_validation import train_test_split


def get_data(pos_file, neg_file, unsup_file):
    """
    load and pretreatment data
    :return: 
    """

    def get_folder_txt(path):
        result = []
        with open(path, 'r', encoding='utf-8') as f:
            result.append(f.read())
        return result

    pos_reviews = get_folder_txt(pos_file)
    neg_reviews = get_folder_txt(neg_file)
    unsup_reviews = get_folder_txt(unsup_file)

    def save_txt(out_path, list):
        with open(out_path, 'w+',encoding='utf-8') as f:
            for i in list:
                f.write(i + '\n')

    save_txt('pos.txt', pos_reviews)
    save_txt('neg.txt', neg_reviews)
    save_txt('unsup.txt', unsup_reviews)
    # 使用1表示正面情感，0为负面
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    # 将数据分割为训练与测试集
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

    # 对英文做简单的数据清洗预处理，中文根据需要进行修改
    def cleanText(corpus):
        punctuation = """.,?!:;(){}[]"""
        corpus = [z.lower().replace('\n', '') for z in corpus]
        corpus = [z.replace('<br />', ' ') for z in corpus]

        # treat punctuation as individual words
        for c in punctuation:
            corpus = [z.replace(c, ' %s ' % c) for z in corpus]
        corpus = [z.split() for z in corpus]
        return corpus

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    unsup_reviews = cleanText(unsup_reviews)

    # Gensim的Doc2Vec应用于训练要求每一篇文章/句子有一个唯一标识的label.
    # 我们使用Gensim自带的TaggedDocument方法. 标识的格式为"TRAIN_i"和"TEST_i"，其中i为序号
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i, v in enumerate(reviews):
            label = '%s_%s' % (label_type, i)
            labelized.append(TaggedDocument(v, [label]))
        return labelized

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')

    return x_train, x_test, unsup_reviews, y_train, y_test


def getVecs(model, corpus, size):
    """
    读取向量
    :param model: 
    :param corpus: 
    :param size: 
    :return: 
    """
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


def train(x_train, x_test, unsup_reviews, size=400, epoch_num=10):
    """
    对数据进行训练
    """
    # 实例DM和DBOW模型
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    # 使用所有的数据建立词典
    model_dm.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
    model_dbow.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))

    # 进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    all_train_reviews = np.concatenate((x_train, unsup_reviews))
    for epoch in range(epoch_num):
        perm = np.random.permutation(all_train_reviews.shape[0])
        model_dm.train(all_train_reviews[perm])
        model_dbow.train(all_train_reviews[perm])

    # 训练测试数据集
    x_test = np.array(x_test)
    for epoch in range(epoch_num):
        perm = np.random.permutation(x_test.shape[0])
        model_dm.train(x_test[perm])
        model_dbow.train(x_test[perm])

    return model_dm, model_dbow


def get_vectors(model_dm, model_dbow):
    """
    将训练完成的数据转换为vectors
    :param model_dm: 
    :param model_dbow: 
    :return: 
    """
    # 获取训练数据集的文档向量
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    # 获取测试数据集的文档向量
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    return train_vecs, test_vecs


def Classifier(train_vecs, y_train, test_vecs, y_test):
    """
    使用分类器对文本向量进行分类训练
    :param train_vecs: 
    :param y_train: 
    :param test_vecs: 
    :param y_test: 
    :return: 
    """
    # 使用sklearn的SGD分类器
    from sklearn.linear_model import SGDClassifier

    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)

    print('Test Accuracy: %.2f' % lr.score(test_vecs, y_test))

    return lr


def ROC_curve(lr, y_test):
    """
    绘出ROC曲线，并计算AUC
    :param lr: 
    :param y_test: 
    :return: 
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    pred_probas = lr.predict_proba(test_vecs)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.show()


##运行模块
if __name__ == "__main__":
    # 设置向量维度和训练次数
    size, epoch_num = 400, 10
    # 获取训练与测试数据及其类别标注
    neg_file = '../data/douban_imdb_data/neg.txt'
    pos_file = '../data/douban_imdb_data/pos.txt'
    unsup_file = '../data/douban_imdb_data/unsup.txt'
    x_train, x_test, unsup_reviews, y_train, y_test = get_data(neg_file, pos_file, unsup_file)
    # 对数据进行训练，获得模型
    model_dm, model_dbow = train(x_train, x_test, unsup_reviews, size, epoch_num)
    # 从模型中抽取文档相应的向量
    train_vecs, test_vecs = get_vectors(model_dm, model_dbow)
    # 使用文章所转换的向量进行情感正负分类训练
    lr = Classifier(train_vecs, y_train, test_vecs, y_test)
    # 画出ROC曲线
    ROC_curve(lr, y_test)
