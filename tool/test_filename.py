# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/15
# Brief: 
with open("1.txt", "r", encoding="utf-8") as f:
    for i in f:
        parts = i.strip().split("\t")
        print(parts[22])
