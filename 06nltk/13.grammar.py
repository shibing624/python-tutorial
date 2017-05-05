# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import nltk
# nltk.data.show_cfg('grammars/book_grammars/sql0.fcfg')
from nltk import load_parser
cp = load_parser('grammars/book_grammars/sql0.fcfg')
query = 'What cities are located in China'
# trees = cp.nbest_parse(query.split())
# answer = trees[0].node['sem']
# q = ''.join(answer)
# print(q)
q = 'SELECT City FROM city_table WHERE Country="china"'
from nltk.sem import chat80
rows = chat80.sql_query('corpora/city_database/city.db',q)
for r in rows:
    print(r[0])