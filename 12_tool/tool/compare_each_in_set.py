# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""


def get_pairs(input_lst):
    out_lst = []
    for i in range(len(input_lst)):
        m = input_lst[i]
        for j in range(i, len(input_lst)):
            n = input_lst[j]
            if m == n: continue
            out_lst.append([m, n])
    return out_lst


lst = ['a','b']
print(get_pairs(lst))
