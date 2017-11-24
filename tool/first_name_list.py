# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/11/2
# Brief: 
import sys
names = set()
with open("first_name_cn.txt",encoding="utf-8",mode="r") as f:
    for line in f:
        lines = line.strip().split(" ")
        for i in lines:
            name = i.split("(")[0]
            names.add(name)

with open("first_name_cn_out.txt",encoding="utf-8",mode="w") as f:
    for i in names:
        f.write(i)
        f.write("\n")