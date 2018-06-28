# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 18/1/25
# Brief: 
import unittest

from feature import Feature
from lr import LR
from xgb import XGB
from xgb_lr import XGBLR
from xgb_util import load_sample_data

train_file = "./data/training_seg_sample.txt"
test_file = "./data/testing_seg_sample.txt"
sep = '\t'
max_feature_cnt = 1000
feature_max_df = 0.95
feature_min_df = 3
ngram_range = (1, 2)
model_path = './data/'
tfidf_model_name = model_path + 'tfidf_feature.model'
best_feature_model_name = model_path + 'best_feature.model'
xgb_model_name = model_path + 'xgb.model'
lr_model_name = model_path + 'lr.model'
xgblr_xgb_model_name = model_path + 'xgblr_xgb.model'
xgblr_lr_model_name = model_path + 'xgblr_lr.model'
one_hot_encoder_model_name = model_path + 'xgblr_ont_hot_encoder.model'


class ClassificationTest(unittest.TestCase):
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
        data_x, data_y = load_sample_data(train_file, sep=sep,has_pos=True)
        self.assertEqual(len(data_x) > 0, True)

    def model_train(self, train_file):
        train_x, train_y = load_sample_data(train_file, sep=sep,has_pos=True)
        features = Feature(tfidf_model_name, best_feature_model_name)
        features.fit(max_feature_cnt, feature_max_df,
                     feature_min_df, ngram_range, train_x, train_y)
        model_train_x_feature = features.transform(train_x)
        # xgboost
        print('train a single xgb model...')
        xgb_clf = LR(xgb_model_name)
        xgb_clf.train_model(model_train_x_feature, train_y)
        print('train a single xgb model done.\n')

        # lr
        print('train a single lr model...')
        lr_clf = LR(lr_model_name)
        lr_clf.train_model(model_train_x_feature, train_y)
        print('train a single LR model done.\n')

        # xgboost+lr
        print('train a xgboost+lr model...')
        xgb_lr_clf = XGBLR(xgblr_xgb_model_name, xgblr_lr_model_name, one_hot_encoder_model_name)
        xgb_lr_clf.train_model(model_train_x_feature, train_y)
        print('train a xgboost+lr model done.\n')

    def model_test(self, test_file):
        test_x, test_y = load_sample_data(test_file, sep=sep,has_pos=True)
        features = Feature(tfidf_model_name, best_feature_model_name)
        features.load_model()
        model_test_x_feature = features.transform(test_x)

        xgb_clf = XGB(xgb_model_name)
        xgb_clf.test_model(model_test_x_feature, test_y)

        lr_clf = LR(lr_model_name)
        lr_clf.test_model(model_test_x_feature, test_y)

        xgb_lr_clf = XGBLR(xgblr_xgb_model_name, xgblr_lr_model_name, one_hot_encoder_model_name)
        xgb_lr_clf.test_model(model_test_x_feature, test_y)

    def test_models(self):
        self.model_train(train_file)
        self.model_test(test_file=test_file)


if __name__ == '__main__':
    unittest.main()
