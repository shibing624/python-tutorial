# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import yaml

with open('domain.yml','r',encoding='utf-8') as f:
    data = yaml.load(f)

print(type(data))
print(data)