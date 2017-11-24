# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/8/30
# Brief: 


import sys

path_user_cdc_client = sys.argv[1]
path_file = sys.argv[2]
path_output = sys.argv[3]

userid_map = {}
with open(path_user_cdc_client, "r")as f:
    for line in f:
        userid = line.strip().split("\t")[0]
        userid = userid.decode("gb18030")
        userid_map[userid] = line.strip().decode("gb18030")

content = set()
with open(path_file, "r") as f:
    for line in f:
        # parts = (line.strip()).decode("utf-8").split("\t")
        # userid = parts[0]
        userid = (line.strip()).decode("utf8")
        if userid in userid_map:
            content.add((line.strip()).decode("utf-8") + "\t" + userid_map[userid])

with open(path_output, "w") as f:
    for line in content:
        f.write((line.strip()).encode("utf-8"))
        f.write("\n")
