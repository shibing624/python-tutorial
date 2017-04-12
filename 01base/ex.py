# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

PI = 3.14


def get_sum(lst):
    """
    Sum the values in the list
    :param lst:
    :return:
    """
    total = 0
    for v in lst:
        total = total + v
    return total


def test():
    l = [1, 2, 3]
    assert (get_sum(l) == 6)
    print("test pass.")


if __name__ == '__main__':
    test()
