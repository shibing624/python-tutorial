# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/9/6
# Brief: 

import sys

path_words = "./normal_word.txt"
words = []
with open(path_words, encoding="utf8") as f:
    for line in f:
        parts = line.strip().split()
        words.extend(parts)

with open("./common_chinese_word.txt", mode="w", encoding="utf-8") as f:
    for i in words:
        if i:
            f.write(i.strip())
            f.write("\n")
