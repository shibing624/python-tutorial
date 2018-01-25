# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 18/1/25
# Brief: 
import pickle
from sklearn.linear_model import LogisticRegression as LR


class LR(object):
    """
    LR model for text classification
    """

    def __init__(self, lr_name):
        self.lr_name = lr_name
        self.init = False

    def train_model(self, train_x, train_y):
        self.clf = LR()
        self.clf.fit(train_x, train_y)
        self.init = True
        pickle.dump(self.clf, self.lr_name, True)

    def load_model(self):
        self.clf = pickle.load(self.lr_name)
        self.init = True

    def test_model(self, test_x, test_y):
        if not self.init:
            self.load_model()

        pred_y = self.clr.predict(test_x)
        total = len(test_y)
        correct = 0
        for idx in range(total):
            if pred_y[idx] == test_y[idx]:
                correct += 1
        print('Test LR:', total, correct, correct * 1.0 / total)
