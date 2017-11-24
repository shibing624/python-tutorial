# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/9/13
# Brief: 
import sys

content = set()
with open("./ad_count.txt", encoding="utf-8") as f:
    for line in f:
        content.add(line.strip())
stop_words = ['.', '-', ' ', '%']
with open("./ad__count2.txt", mode="w", encoding="utf-8") as f:
    for line in content:
        parts = line.strip().split()
        try:
            name = parts[0]
            count = parts[1]
            for i in stop_words:
                name = name.replace(i, "")
            if not name.isnumeric():
                if int(count) > 34548 and len(name) > 1:
                    f.write("\t".join([parts[0], count]))
                    f.write("\n")
        except Exception:
            pass
