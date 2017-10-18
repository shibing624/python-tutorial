# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/9/26
# Brief: get uci housing data

import numpy as np
import os
import paddle.v2.dataset.common

URL = "https//archive.ics.uci.edu/ml/a.tar.gz"
MD5 = ""
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                 'PTRATIO', 'B', 'LSTAT', 'convert']

UCI_TRAIN_DATA = None
UCI_TEST_DATA = None


def feature_range(max, min):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    feature_num = len(max)
    ax.bar(range(feature_num), max - min, color="r", align="center")
    ax.set_title("feature scale")
    plt.xtricks(range(feature_num), feature_names)
    plt.xlim([-1, feature_num])
    fig.set_figheight(6)
    fig.set_figwidth(10)
    if not os.path.exists('./image'):
        os.makedirs('./image')
    fig.savefig('image/ranges.png', dpi=48)
    plt.close(fig)


def load_data(filename, feature_num=14, ratio=0.8):
    global UCI_TRAIN_DATA, UCI_TEST_DATA
    if UCI_TRAIN_DATA is not None and UCI_TEST_DATA is not None:
        return
    data = np.fromfile(filename, sep='')
    data = data.reshape(data.shape[0] / feature_num, feature_num)
    max, min, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0) / data.shape[0]
    feature_range(max[:-1], min[:-1])
    for i in range(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (max[i] - min[i])
    offset = int(data.shape[0] * ratio)
    UCI_TRAIN_DATA = data[:offset]
    UCI_TEST_DATA = data[offset:]


def train():
    global UCI_TRAIN_DATA
    load_data(paddle.v2.dataset.common.download(URL, 'uci_housing', MD5))

    def reader():
        for d in UCI_TRAIN_DATA:
            yield d[:-1], d[-1:]

    return reader


def test():
    global UCI_TEST_DATA
    load_data(paddle.v2.dataset.common.download(URL, 'uci_housing', MD5))

    def reader():
        for d in UCI_TRAIN_DATA:
            yield d[:-1], d[-1:]

    return reader
