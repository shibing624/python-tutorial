# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 18/1/25
# Brief: 
import pickle

from sklearn.linear_model import LogisticRegression


class LR(object):
    """
    LR model for text classification
    """

    def __init__(self, lr_name):
        self.lr_name = lr_name
        self.init = False

    def train_model(self, train_x, train_y):
        self.clf = LogisticRegression()
        self.clf.fit(train_x, train_y)
        self.init = True
        with open(self.lr_name, 'wb') as f:
            pickle.dump(self.clf, f, True)

    def load_model(self, lr_name):
        with open(lr_name, 'rb') as f:
            self.clf = pickle.load(f)
            self.init = True

    def test_model(self, test_x, test_y):
        if not self.init:
            print("not init lr model. load ...")
            self.load_model(self.lr_name)
            print("load lr model done.")

        pred_y = self.clf.predict(test_x)
        total = len(test_y)
        correct = 0
        for idx in range(total):
            if pred_y[idx] == test_y[idx]:
                correct += 1
        print('Test LR: total_count, right_count, pred:', total, correct, correct * 1.0 / total)
        return pred_y