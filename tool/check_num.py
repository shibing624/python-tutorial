# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/8/29
# Brief: 



import sys

file1 = sys.argv[1]  # watch file
file2 = sys.argv[2]  # repair file
file_result = sys.argv[3]
file1_set = set()
file2_set = set()
with open(file1, encoding="utf-8")as f1:
    for line in f1:
        line = line.strip()
        file1_set.add(line)
with open(file2, encoding="utf-8") as f2:
    for line in f2:
        line = line.strip()
        file2_set.add(line)

with open(file_result, encoding="utf-8", mode="a") as wf:
    for i in file2_set:
        if i in file1_set:
            wf.write(i)
            wf.write('\t')
            wf.write('watch')
            wf.write('\n')
        else:
            wf.write(i)
            wf.write('\t')
            wf.write('repair')
            wf.write('\n')

    count = 0
    for i in file1_set:
        if i not in file2_set:
            wf.write(i)
            wf.write('\t')
            wf.write('repair')
            wf.write('\n')
            count = 1 + count
    print(count)
