# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/16
# Brief: 数据处理

import os
import re

import jieba
import numpy as np


def clean_str(string):
    """
    Tokenization cleaning for dataset, only save english chars
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


def contain_chinese(string):
    """
    Check string is contain chinese character
    :param string:
    :return:
    """
    zh_pattern = re.compile(r"[\u4E00-\u9FD5]+")
    return zh_pattern.search(string)


def get_segment(sent):
    clean_text = []
    # 中文
    if contain_chinese(sent):
        # 用1元切分
        # clean_text.append(" ".join(list(sent)))
        # jieba切词
        clean_text.append(" ".join(jieba.cut(sent)))
    else:
        # 英文用clean_str切分
        clean_text.append(clean_str(sent))
    return clean_text


def load_data_labels(data_dir):
    """
    Loads polarity data from files, splits data to words and labels,
    Split chinese word with
    :param data_dir:
    :return: split sentence and labels
    """
    words = []
    labels = []
    for file_name in os.listdir(data_dir):
        with open(os.path.join(data_dir, file_name), mode='r', encoding='utf-8') as f:
            for line in f:
                # label in first sep
                parts = line.strip().split(',', 1)
                if parts and len(parts) > 1:
                    # label start with 0
                    lbl = int(parts[0])
                    sent = parts[1]
                    sent_split = " ".join(jieba.cut(sent))
                    words.append(sent_split)
                    labels.append(lbl)
    return [words, labels]


def load_infer_data(lines):
    """
    Loads infer data
    :param positive_data_file:
    :param negative_data_file:
    :return: split sentence and labels
    """
    x_text = [s.strip() for s in lines]
    clean_text = []
    for sent in x_text:
        # 中文
        if contain_chinese(sent):
            # 用1元切分
            # clean_text.append(" ".join(list(sent)))
            # jieba切词
            clean_text.append(" ".join(jieba.cut(sent)))
        else:
            # 英文用clean_str切分
            clean_text.append(clean_str(sent))
    return clean_text


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
