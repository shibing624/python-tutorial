# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

from feature import Feature
from lr import LR
from xgb_lr import XGBLR
from xgb_util import load_sample_data

train_file = "./data/training_seg.txt"
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


def model_train(train_file):
    train_x, train_y = load_sample_data(train_file, sep=sep, has_pos=True)
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

if __name__ == "__main__":
    model_train(train_file=train_file)

