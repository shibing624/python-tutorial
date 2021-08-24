# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 18/1/25
# Brief: 
import pickle

import numpy as np
import xgboost as xgb


class XGB(object):
    """
    xgboost model for text classification
    """

    def __init__(self, xgb_model_name,
                 eval_metric='auc'):
        self.eval_metric = eval_metric
        self.xgb_model_name = xgb_model_name
        self.init = False

    def train_model(self, train_x, train_y):
        """
        use Feature vector
        :param train_x:
        :param train_y:
        :return:
        """
        self.clf = xgb.XGBClassifier()
        self.clf.fit(train_x, train_y, eval_metric=self.eval_metric,
                     eval_set=[(train_x, train_y)])
        self.init = True
        evals_result = self.clf.evals_result()
        print('evals_result:', evals_result)
        with open(self.xgb_model_name, 'wb')as f:
            pickle.dump(self.clf, f, True)

    def load_model(self, xgb_model_name):
        with open(xgb_model_name, 'rb') as f:
            self.clf = pickle.load(f)
            self.init = True

    def test_model(self, test_x, test_y):
        if not self.init:
            print("not init xgb model. load ...")
            self.load_model(self.xgb_model_name)
            print("load xgb model done.")
        pred_x = self.clf.predict_proba(test_x)
        pred_idx_arr = np.argmax(pred_x, axis=1)
        pred_label = [self.clf.classes_[idx] for idx in pred_idx_arr]

        total = len(test_y)
        correct = 0
        for idx in range(total):
            if pred_label[idx] == test_y[idx]:
                correct += 1
        print('Xgb test:', total, correct, correct * 1.0 / total)
        return pred_label