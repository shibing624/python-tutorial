# from data_utils import get_stock_data
import os
import time

import numpy as np
import pandas as pd
import tushare as ts

from config import Arg

args = Arg()


# 该系列代码所要求的股票文件名称必须是股票代码+csv的格式，如000001.csv
# --------------------------训练集数据的处理--------------------- #
def get_train_data(batch_size=args.batch_size, time_step=args.time_step):
    ratio = args.ratio
    stock_len = args.stock_len
    len_index = []
    batch_index = []
    val_index = []
    train_dir = args.train_dir
    df = open(train_dir)
    data_otrain = pd.read_csv(df)
    data_train = data_otrain.iloc[:, 1:].values
    print(len(data_train))
    label_train = data_otrain.iloc[:, -1].values
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    train_x, train_y = [], []  # 训练集x和y定义
    for i in range(len(normalized_train_data) + 1):
        if i % stock_len == 0:
            len_index.append(i)
    for i in range(len(len_index) - 1):
        for k in range(len_index[i], len_index[i + 1] - time_step - 1):
            x = normalized_train_data[k:k + time_step, :6]
            y = label_train[k + time_step, np.newaxis]
            temp_data = []
            # onehot编码
            for j in y:
                if j > 2:
                    temp_data.append([0, 0, 0, 0, 0, 1])
                elif 1 < j <= 2:
                    temp_data.append([0, 0, 0, 0, 1, 0])
                elif 0 < j <= 1:
                    temp_data.append([0, 0, 0, 1, 0, 0])
                elif -1 < j <= 0:
                    temp_data.append([0, 0, 1, 0, 0, 0])
                elif -2 < j <= -1:
                    temp_data.append([0, 1, 0, 0, 0, 0])
                else:
                    temp_data.append([1, 0, 0, 0, 0, 0])
            train_x.append(x.tolist())
            train_y.append(temp_data)
    train_len = int(len(train_x) * ratio)  # 按照8：2划分训练集和验证集
    train_x_1, train_y_1 = train_x[:train_len], train_y[:train_len]  # 训练集的x和标签
    val_x, val_y = train_x[train_len:], train_y[train_len:]  # 验证集的x和标签
    # 添加标签
    for i in range(len(train_x_1)):
        if i % batch_size == 0:
            batch_index.append(i)
    for i in range(len(val_x)):
        if i % batch_size == 0:
            val_index.append(i)
    batch_index.append(len(train_x_1))
    val_index.append(len(val_x))
    print(batch_index)
    print(val_index)
    print(np.shape(train_x))
    return batch_index, val_index, train_x_1, train_y_1, val_x, val_y


# --------------------------测试集数据的处理--------------------- #
# 测试集数据长度不能小于time_step
def get_test_data(time_step=args.time_step):
    stock_len = args.stock_len
    test_dir = args.test_dir
    f = open(test_dir)
    df = pd.read_csv(f)
    data_test = df.iloc[:, 1:].values
    label_test = df.iloc[:, -1].values
    batch_index = []
    normalized_test_data = (data_test - np.mean(data_test, axis=0)) / np.std(data_test, axis=0)  # 标准化
    test_x, test_y = [], []
    for i in range(len(normalized_test_data) + 1):
        if i % stock_len == 0:
            batch_index.append(i)
    for i in range(len(batch_index) - 1):
        if stock_len > time_step + 1:
            for j in range(batch_index[i], batch_index[i + 1] - time_step - 1):
                x = normalized_test_data[j:j + time_step, :]
                y = label_test[j + time_step, np.newaxis]
                temp_data = []
                # 标签编码
                for k in y:
                    if k > 2:
                        temp_data.append([0, 0, 0, 0, 0, 1])
                    elif 1 < k <= 2:
                        temp_data.append([0, 0, 0, 0, 1, 0])
                    elif 0 < k <= 1:
                        temp_data.append([0, 0, 0, 1, 0, 0])
                    elif -1 < k <= 0:
                        temp_data.append([0, 0, 1, 0, 0, 0])
                    elif -2 < k <= -1:
                        temp_data.append([0, 1, 0, 0, 0, 0])
                    else:
                        temp_data.append([1, 0, 0, 0, 0, 0])
                test_x.append(x.tolist())
                test_y.extend(temp_data)
        else:
            for j in range(batch_index[i], batch_index[i] + 1):
                x = normalized_test_data[j:j + time_step, :]
                y = label_test[j + time_step, np.newaxis]
                temp_data = []
                # 标签编码
                for k in y:
                    if k > 2:
                        temp_data.append([0, 0, 0, 0, 0, 1])
                    elif 1 < k <= 2:
                        temp_data.append([0, 0, 0, 0, 1, 0])
                    elif 0 < k <= 1:
                        temp_data.append([0, 0, 0, 1, 0, 0])
                    elif -1 < k <= 0:
                        temp_data.append([0, 0, 1, 0, 0, 0])
                    elif -2 < k <= -1:
                        temp_data.append([0, 1, 0, 0, 0, 0])
                    else:
                        temp_data.append([1, 0, 0, 0, 0, 0])
                test_x.append(x.tolist())
                test_y.extend(temp_data)

    print(batch_index)
    print(np.shape(test_x))
    return test_x, test_y


# -------------------------新股票数据批量处理-------------------- #
# 该函数将更新数据加历史数据的前time_step-1拼接,用于整批处理
# 如2019 1-3月数据和2018.12的数据拼接
def get_update_data(time_step=args.time_step):
    train_data_dir = args.train_dir
    new_data_dir = args.new_dir
    stock_len = args.stock_len
    new_len = args.stock_len_new
    f = open(train_data_dir)
    nf = open(new_data_dir)
    df = pd.read_csv(f)
    ndf = pd.read_csv(nf)
    data_train = df.iloc[:, 1:].values
    data_new = ndf.iloc[:, 1:].values
    # 标准化
    mean, std = np.mean(data_train, axis=0), np.std(data_train, axis=0)
    new_mean, new_std = np.mean(data_new, axis=0), np.std(data_new, axis=0)
    normalized_data_train = (data_train - mean) / std
    normalized_data_new = (data_new - new_mean) / new_std
    label_new = ndf.iloc[:, -1].values
    train_x, train_y = [], []
    batch_index = []
    new_index = []
    for i in range(len(data_train) + 1):
        if i % stock_len == 0:
            batch_index.append(i)
    for i in range(len(data_new) + 1):
        if i % new_len == 0:
            new_index.append(i)
    # 该部分添加上一时间戳和更新时间戳的time_step股票数据
    print(batch_index)
    print(new_index)
    for i in range(1, len(batch_index)):
        count = time_step
        while count > 1:
            last_data = []
            last_data.extend(normalized_data_train[batch_index[i] - count + 1:batch_index[i]])
            last_data.extend(normalized_data_new[0:time_step - count + 1])
            y = label_new[time_step - count + 1, np.newaxis]
            temp_data = []
            for k in y:
                if k > 2:
                    temp_data.append([0, 0, 0, 0, 0, 1])
                elif 1 < k <= 2:
                    temp_data.append([0, 0, 0, 0, 1, 0])
                elif 0 < k <= 1:
                    temp_data.append([0, 0, 0, 1, 0, 0])
                elif -1 < k <= 0:
                    temp_data.append([0, 0, 1, 0, 0, 0])
                elif -2 < k <= -1:
                    temp_data.append([0, 1, 0, 0, 0, 0])
                else:
                    temp_data.append([1, 0, 0, 0, 0, 0])
            train_x.append(last_data)
            train_y.append(temp_data)
            count -= 1
    # 该部分正常添加new_data中的股票数据
    for i in range(len(new_index) - 1):
        for j in range(new_index[i], new_index[i + 1] - time_step - 1):
            x = normalized_data_new[j:j + time_step, :]
            y = label_new[j + time_step, np.newaxis]
            temp_data = []
            # 标签编码
            for k in y:
                if k > 2:
                    temp_data.append([0, 0, 0, 0, 0, 1])
                elif 1 < k <= 2:
                    temp_data.append([0, 0, 0, 0, 1, 0])
                elif 0 < k <= 1:
                    temp_data.append([0, 0, 0, 1, 0, 0])
                elif -1 < k <= 0:
                    temp_data.append([0, 0, 1, 0, 0, 0])
                elif -2 < k <= -1:
                    temp_data.append([0, 1, 0, 0, 0, 0])
                else:
                    temp_data.append([1, 0, 0, 0, 0, 0])
            train_x.append(x.tolist())
            train_y.append(temp_data)
    print(np.shape(train_x))
    return train_x, train_y


# --------------------------当天股票数据更新---------------------- #
# 该函数完成下载实时股票数据，与之前的数据拼接后拼接的x
# 只能用于获取一天的更新数据，不会对源文件进行更新，如果有断层（不只一天），请先下载整批数据，然后使用get_update_data来更新数据
# file_name是要用于预测的股票地址如'D:\data\\201904\\000001.csv'
def get_predict_data(file_name):
    f = open(file_name)
    (filepath, temp_file_name) = os.path.split(file_name)
    (stock_code, extension) = os.path.splitext(temp_file_name)
    f = pd.read_csv(f)
    data_now = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    hist_data = f[-args.time_step + 1:]
    real_data = ts.get_realtime_quotes(stock_code)
    real_data = real_data[['open', 'high', 'price', 'low', 'volume']]
    real_data.insert(0, 'date', data_now)
    p_change = (float(real_data.iloc[-1, 3]) - float(hist_data.iloc[-1, 3])) / float(hist_data.iloc[-1, 3])
    real_data['p_change'] = p_change
    real_data.rename(index=str, columns={"price": "close"}, inplace=True)
    real_data[['open', 'high', 'close', 'low', 'volume', 'p_change']] = \
        real_data[['open', 'high', 'close', 'low', 'volume', 'p_change']].astype('float')
    hist_data = hist_data.append(real_data)
    print("---------数据更新完成-----------")
    pre_data = hist_data.iloc[:, 1:].values
    x = (pre_data - np.mean(pre_data, axis=0)) / np.std(pre_data, axis=0)  # 标准化
    x = [x.tolist()]
    print(np.shape(x))
    return x, stock_code


if __name__ == '__main__':
    print(args.stock_len)
    get_train_data()
    # get_stock_data('399300', '2019-04-01', '2019-04-30', './data/')
    # get_update_data()
    #  get_stock_data()使用样例，用于下载codelist中的股票，codelist可以自己指定
"""
    codelist = []
    with open('d:\data\codelist.csv') as f:
        df = pd.read_csv(f, converters={'code': str})
        codelist.extend(df['code'])
    i = 1
    for code in codelist:
        print('正在处理第%s个股票' % i)
        i += 1
        get_stock_data(code, '2019-04-01', '2019-04-30', 'd:\data\\201904\\', 20)
"""
