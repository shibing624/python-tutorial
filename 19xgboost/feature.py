# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 18/1/25
# Brief: 
import sys
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class Feature(object):
    """
    select features from the raw text
    """

    def __init__(self, feature_vec_name, best_feature_name):
        self.feature_vec_name = feature_vec_name
        self.best_feature_name = best_feature_name
        self.init = False

    def fit_model(self, train_x, train_y):
        best_k = self.max_feature_cnt
        vec_max_df = self.feature_max_df
        vec_min_df = self.feature_min_df
        vec_ngram_range = self.ngram_range
        self.tf_vec = TfidfVectorizer(ngram_range=vec_ngram_range,
                                      min_df=vec_min_df, max_df=vec_max_df)
        self.best = SelectKBest(chi2, k=best_k)
        train_tf_vec = self.tf_vec.fit_transform(train_x)

    def set_feature_para(self, max_feature_cnt, feature_max_df,
                         feature_min_df, ngram_range):
        self.max_feature_cnt = max_feature_cnt
        self.feature_max_df = feature_max_df
        self.feature_min_df = feature_min_df
        self.ngram_range = ngram_range

    def fit(self, max_feature_cnt, feature_max_df,
            feature_min_df, ngram_range, train_x, train_y):
        self.set_feature_para(max_feature_cnt, feature_max_df,
                              feature_min_df, ngram_range)
        self.fit_model(train_x, train_y)

        pickle.dump(self.tf_vec, self.feature_vec_name, True)
        pickle.dump(self.best, self.best_feature_name, True)

    def load_model(self):
        self.tf_vec = pickle.load(self.feature_vec_name)
        self.best = pickle.load(self.best_feature_name)
        self.init = True

    def transform(self, x_test):
        if not self.init:
            self.load_model()
        x_vec = self.tf_vec.transform(x_test)
        x_best = self.best.transform(x_vec)
        return x_best
