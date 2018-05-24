# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

import sys

type = sys.getfilesystemencoding()


def eclat(prefix, items, min_support, freq_items):
    while items:
        # 初始遍历单个的元素是否是频繁
        key, item = items.pop()
        key_support = len(item)
        if key_support >= min_support:
            # print frozenset(sorted(prefix+[key]))
            freq_items[frozenset(sorted(prefix + [key]))] = key_support
            suffix = []  # 存储当前长度的项集
            for other_key, other_item in items:
                new_item = item & other_item  # 求和其他集合求交集
                if len(new_item) >= min_support:
                    suffix.append((other_key, new_item))
            eclat(prefix + [key], sorted(suffix, key=lambda item: len(item[1]), reverse=True), min_support, freq_items)
    return freq_items


def eclat_zc(data_set, min_support=1):
    """
    Eclat方法
    :param data_set:
    :param min_support:
    :return:
    """
    # 将数据倒排
    data = {}
    trans_num = 0
    for trans in data_set:
        trans_num += 1
        for item in trans:
            if item not in data:
                data[item] = set()
            data[item].add(trans_num)
    freq_items = {}
    freq_items = eclat([], sorted(data.items(), key=lambda item: len(item[1]), reverse=True), min_support, freq_items)
    return freq_items

if __name__ == "__main__":
    print('hello')