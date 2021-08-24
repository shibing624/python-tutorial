# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 18/1/25
# Brief: 

def load_sample_data(data_path, sep=";", has_pos=False):
    """
    load data by segmented corpus
    :param data_path:
    :return:
    """
    data_x = []
    data_y = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep, 1)
            if len(parts) < 2:
                print('err, must more than 2 parts.')
                continue
            text = parts[1]
            if has_pos:
                words = trim_pos(text)
            else:
                words = parts[1]
            data = ' '.join(words)
            tag = parts[0].strip()
            if tag == '':
                continue
            data_x.append(data)
            data_y.append(tag)
    return data_x, data_y


def trim_pos(text):
    word_pos_list = text.split(' ')
    word_list = [w.split('/')[0] for w in word_pos_list]
    return word_list


def load_data(data_path, sep='\t'):
    """
        load data with features
        :param data_path:
        :return:
        """
    data = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if len(parts) < 2:
                print('err, must more than 2 parts.')
                continue
            data.append([float(i) for i in parts])
    return data


def load_lr_data(data_path, sep='\t'):
    """
        load data with features
        :param data_path:
        :return:
        """
    data_x = []
    data_y = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if len(parts) < 2:
                print('err, must more than 2 parts.')
                continue

            data_x.append([float(i) for i in parts[1:]])
            data_y.append(float(parts[0]))
    return data_x, data_y
