# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report


class XGB(object):
    """
    xgboost model for text classification
    """

    def __init__(self, xgb_model_name):
        self.xgb_model_name = xgb_model_name
        self.init = False

    def train_model(self, train_data):
        """
        use Feature vector
        :param train_x:
        :param train_y:
        :return:
        """
        data_np = np.array(train_data, dtype=float)
        data_pd = pd.DataFrame(data=data_np)
        train = data_pd.iloc[:, 1:].values
        labels = data_pd.iloc[:, :1].values

        # Using 10000 rows for early stopping.
        offset = 500  # 训练集中数据5000，划分4500用作训练，500用作验证
        # 划分训练集与验证集
        xgtrain = xgb.DMatrix(train[:offset, :], label=labels[:offset])
        xgval = xgb.DMatrix(train[offset:, :], label=labels[offset:])

        params = {
            'objective': 'binary:logistic',
            'gamma': 0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
            'max_depth': 12,  # 构建树的深度 [1:]
            'subsample': 0.8,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
            'colsample_bytree': 0.8,  # 构建树树时的采样比率 (0:1]
            'silent': 1,
            'eta': 0.01,  # 如同学习率
            'seed': 10,
        }
        plst = list(params.items())
        num_rounds = 100  # 迭代你次数
        # return 训练和验证的错误率
        watchlist = [(xgtrain, 'train'), (xgval, 'val')]
        # training model
        # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
        self.clf = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

        with open(self.xgb_model_name, 'wb')as f:
            pickle.dump(self.clf, f, True)

    def load_model(self, xgb_model_name):
        with open(xgb_model_name, 'rb') as f:
            self.clf = pickle.load(f)
            self.init = True

    def test_model(self, test_data):
        if not self.init:
            print("not init xgb model. load ...")
            self.load_model(self.xgb_model_name)
            print("load xgb model done.")

        data_np = np.array(test_data, dtype=float)
        data_pd = pd.DataFrame(data=data_np)
        test = data_pd.iloc[:, 1:].values
        labels = data_pd.iloc[:, :1].values

        xgtest = xgb.DMatrix(test)
        preds = self.clf.predict(xgtest)
        print('pred:', preds[:10])
        total_count = int(len(preds))
        right_count = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) == labels[i])
        out_preds = []
        for i in range(len(preds)):
            if float(preds[i]) > 0.5:
                out_preds.append(1)
            else:
                out_preds.append(0)
        target_names = ['risk', 'norisk']
        report = classification_report(labels, out_preds, target_names=target_names)
        print(report)

        print('Xgb test: total_count, right_count, pred:', total_count, right_count, right_count / total_count)
