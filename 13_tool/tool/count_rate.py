# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/15
# Brief: 
import sys

path_out = sys.argv[1]
count_total = 0
word_total = 0
idea_total = 0
word_right = 0
idea_right = 0

word_wrong_set = set()
idea_wrong_set = set()
for line in sys.stdin:
    parts = line.strip().split("\t")
    m_type = parts[3]

    if m_type == 'word':
        word_total += 1
        word_status = parts[20]
        if word_status == 'APPROVED':
            word_right += 1
        else:
            word_wrong_set.add(line.strip())
    elif m_type == 'idea':
        idea_total += 1
        idea_status = parts[20]
        if idea_status == 'APPROVED':
            idea_right += 1
        else:
            idea_wrong_set.add(line.strip())

word_rate = word_right / word_total
idea_rate = idea_right / idea_total
with open(path_out, 'w') as f:
    f.write("word_rate:" + str(word_rate) + "\n")
    f.write("idea_rate:" + str(idea_rate) + "\n")
    f.write("\n\n")
    f.write("word wrong line:\n")
    for word_line in word_wrong_set:
        f.write(word_line + "\n")
    f.write("\nidea wrong line:\n")
    for idea_line in idea_wrong_set:
        f.write(idea_line + "\n")
