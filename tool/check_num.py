# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/8/29
# Brief: 



import sys

file1 = sys.argv[1]  # black file
file2 = sys.argv[2]  # out file

file1_set = set()
file2_set = set()
with open(file1, encoding="utf-8")as f1:
    for line in f1:
        line = line.strip()
        parts = line.split("|")
        for i in parts:
            file1_set.add(i)

with open(file2, encoding="utf-8", mode="w") as f2:
    for i in file1_set:
        f2.write(i + "\n")
