# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import pycorrector

with open('eng_chi.txt', encoding='utf-8') as f1, open('a.txt', 'w', encoding='utf-8')as f2:
    for line in f1:
        line = line.strip()
        parts = line.split('\t')
        eng = parts[0]
        chi = parts[1]
        f2.write('src: ' + eng + "\n")
        f2.write('dst: ' + pycorrector.traditional2simplified(chi) + '\n')
