# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 18/1/25
# Brief: 
import unittest
from util import load_data, load_lr_data
from lr import LR
from basic_xgb import XGB

train_file = "./data/train.demo.txt"
test_file = "./data/test.demo.txt"
ngram_range = (1, 2)
model_path = './data/'
xgb_model_name = model_path + 'xgb.model'
lr_model_name = model_path + 'lr.model'


class ATest(unittest.TestCase):
    """Test Case for classification
    """

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_init(self):
        print("test_init")
        """测试初始化函数，捕捉异常"""
        data_x = load_data(train_file)
        self.assertEqual(len(data_x) > 0, True)

    def model_train(self, train_file):
        train_data = load_data(train_file)
        # xgboost
        print('train a single xgb model...')
        xgb_clf = XGB(xgb_model_name)
        xgb_clf.train_model(train_data)
        print('train a single xgb model done.\n')

        # lr
        print('train a single lr model...')
        lr_clf = LR(lr_model_name)
        x, y = load_lr_data(train_file, '\t')
        lr_clf.train_model(x, y)
        print('train a single LR model done.\n')

    def model_test(self, test_file):
        test_data = load_data(test_file)

        xgb_clf = XGB(xgb_model_name)
        xgb_clf.test_model(test_data)

        lr_clf = LR(lr_model_name)
        x, y = load_lr_data(test_file, '\t')
        lr_clf.test_model(x, y)

    def test_models(self):
        self.model_train(train_file)
        self.model_test(test_file)


if __name__ == '__main__':
    unittest.main()
