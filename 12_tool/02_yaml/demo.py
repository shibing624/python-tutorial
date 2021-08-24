# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import yaml

with open('a.yml','r') as f:
    data = yaml.load(f)

print(type(data))
print(data)