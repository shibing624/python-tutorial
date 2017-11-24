# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/9/18
# Brief: 
import sys

tag_map = {}
with open("taguser.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if parts and float(parts[2]) >= 0.95:
            key, val = parts[0], parts[1]
            if key and val:
                tag_map[key] = val

common_ad_words = set()
with open("ad_words.txt", encoding="utf-8")as f:
    for line in f:
        common_ad_words.add(line.strip().strip("\t")[0])

text_set = set()
with open("demo.txt", encoding="utf-8")as f:
    for line in f:
        parts = line.strip().split("\t")
        userid = parts[4]
        variant_word = parts[-2]
        if variant_word not in common_ad_words and userid in tag_map:
            text_set.add("\t".join([line.strip(), "tag:", tag_map[userid]]))

with open("result.txt", encoding="utf-8", mode="w")as f:
    for line in text_set:
        f.write(line)
        f.write("\n")
