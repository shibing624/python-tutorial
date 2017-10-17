# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/16
# Brief: 数据处理

import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization cleaning for dataset
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param string:
    :return:
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_labels(positive_data_file, negative_data_file):
    """
    Loads polarity data from files, splits data to words and labels
    :param positive_data_file:
    :param negative_data_file:
    :return: split sentence and labels
    """
    positive_data = list(open(positive_data_file, "r", encoding="utf-8").readlines())
    positive_data = [s.strip() for s in positive_data]
    negative_data = list(open(negative_data_file, "r", encoding="utf-8").readlines())
    negative_data = [s.strip() for s in negative_data]
    # split by words
    x_text = positive_data + negative_data
    x_text = [clean_str(sent) for sent in x_text]
    # generate labels
    positive_labels = [[0, 1] for i in positive_data]
    negative_labels = [[1, 0] for i in negative_data]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generate a batch iterator for dataset
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return: batch iterator
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]
