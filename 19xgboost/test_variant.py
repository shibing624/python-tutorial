# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 18/1/25
# Brief: 
import unittest
from util import load_variant_data
from feature import Feature
from lr import LR
from xgb import XGB
from xgb_lr import XGBLR

train_file = "./data/train.demo.txt"
test_file = "./data/test.demo.txt"
max_feature_cnt = 40
feature_max_df = 0.55
feature_min_df = 3
ngram_range = (1, 2)
model_path = './data/'
xgb_model_name = model_path + 'xgb.model'
lr_model_name = model_path + 'lr.model'
xgblr_xgb_model_name = model_path + 'xgblr_xgb.model'
xgblr_lr_model_name = model_path + 'xgblr_lr.model'
one_hot_encoder_model_name = model_path + 'xgblr_ont_hot_encoder.model'


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
        data_x, data_y = load_variant_data(train_file)
        self.assertEqual(len(data_x) > 0, True)

    def model_train(self, train_file):
        train_x, train_y = load_variant_data(train_file)
        # xgboost
        print('train a single xgb model...')
        xgb_clf = LR(xgb_model_name)
        xgb_clf.train_model(train_x, train_y)
        print('train a single xgb model done.\n')

        # lr
        print('train a single lr model...')
        lr_clf = LR(lr_model_name)
        lr_clf.train_model(train_x, train_y)
        print('train a single LR model done.\n')

    def model_test(self, test_file):
        test_x, test_y = load_variant_data(test_file)

        xgb_clf = XGB(xgb_model_name)
        xgb_clf.test_model(test_x, test_y)

        lr_clf = LR(lr_model_name)
        lr_clf.test_model(test_x, test_y)

    def test_models(self):
        self.model_train(train_file)
        self.model_test(test_file)


if __name__ == '__main__':
    unittest.main()
