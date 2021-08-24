# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

from feature import Feature
from lr import LR
from xgb import XGB
from xgb_lr import XGBLR
from xgb_util import load_sample_data

test_file = "./data/validation_seg.txt"
sep = '\t'
model_path = './data/'
tfidf_model_name = model_path + 'tfidf_feature.model'
best_feature_model_name = model_path + 'best_feature.model'
xgb_model_name = model_path + 'xgb.model'
xgb_pred_name = model_path + 'xgb_pred.txt'
lr_model_name = model_path + 'lr.model'
lr_pred_name = model_path + 'lr_pred.txt'
xgblr_xgb_model_name = model_path + 'xgblr_xgb.model'
xgblr_lr_model_name = model_path + 'xgblr_lr.model'
xgblr_pred_name = model_path + 'xgblr_lr_pred.txt'
one_hot_encoder_model_name = model_path + 'xgblr_ont_hot_encoder.model'

label_revserv_dict = {0: '人类作者',
                      1: '机器作者',
                      2: '机器翻译',
                      3: '自动摘要'}


def model_test(test_file):
    test_x, test_y = load_sample_data(test_file, sep=sep, has_pos=True)
    features = Feature(tfidf_model_name, best_feature_model_name)
    features.load_model()
    model_test_x_feature = features.transform(test_x)

    xgb_clf = XGB(xgb_model_name)
    xgb_preds = xgb_clf.test_model(model_test_x_feature, test_y)

    lr_clf = LR(lr_model_name)
    lr_preds = lr_clf.test_model(model_test_x_feature, test_y)

    xgb_lr_clf = XGBLR(xgblr_xgb_model_name, xgblr_lr_model_name, one_hot_encoder_model_name)
    xgb_lr_preds = xgb_lr_clf.test_model(model_test_x_feature, test_y)

    save(xgb_preds, pred_save_path=xgb_pred_name)
    save(lr_preds, pred_save_path=lr_pred_name)
    save(xgb_lr_preds, pred_save_path=xgblr_pred_name)


def save(label_pred, test_ids=[], pred_save_path=None):
    if pred_save_path:
        with open(pred_save_path, 'w', encoding='utf-8') as f:
            for i in range(len(label_pred)):
                if test_ids and len(test_ids) > 0:
                    assert len(test_ids) == len(label_pred)
                    f.write(str(test_ids[i]) + ',' + label_revserv_dict[label_pred[i]] + '\n')
                else:
                    f.write(str(label_pred[i]) + ',' + label_revserv_dict[int(label_pred[i])] + '\n')
        print("pred_save_path:", pred_save_path)


if __name__ == '__main__':
    model_test(test_file=test_file)
