# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 18/1/25
# Brief: 

def load_sample_data(data_path, sep=";"):
    """
    load data by segmented corpus
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
            data = ' '.join(parts[1:])
            tag = parts[0].strip()
            if tag == '':
                continue
            data_x.append(data)
            data_y.append(tag)
    return data_x, data_y


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
