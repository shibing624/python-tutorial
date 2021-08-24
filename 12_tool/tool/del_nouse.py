# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

common_char_set = set()
with open('common_char_set.txt', 'r', encoding='utf-8')as f:
    for line in f:
        common_char_set.add(line.strip())

inputs = []
with open('a.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        inputs.append(line)

with open('aa.txt', 'w', encoding='utf-8') as f:
    for line in inputs:
        out = ''
        for i in line.strip():
            if i == '\t':
                out += i
            if i in common_char_set:
                out += i

        f.write(out + '\n')
