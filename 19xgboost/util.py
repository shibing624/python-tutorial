# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 18/1/25
# Brief: 
import sys


def load_load(data_path):
    """
    load data by segmented corpus
    :param data_path:
    :return:
    """
    data_x = []
    data_y = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print('err, must be 2 parts.')
                continue
            data = ' '.join(parts[1:])
            tag = parts[0].strip()
            if tag == '':
                continue
            data_x.append(data)
            data_y.append(tag)
    return data_x, data_y
