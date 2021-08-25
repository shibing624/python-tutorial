# -*- coding: gb18030 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/8/30
# Brief: 

import sys

file1 = sys.argv[1]  # watch file
with open(file1, 'r')as f1:
    for line in f1:
        try:
            parts = line.strip().split("\t")
            if parts[3] != '4':
                print("\t".join([parts[0], parts[1], parts[2], parts[3], parts[9]]))
        except Exception:
            pass
