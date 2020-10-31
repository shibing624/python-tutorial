# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import yaml
data = ''
with open('yaml_demo.yaml', 'r') as f:
    content = yaml.safe_load(f)
    print(content)
    data = content

new_file = "yaml_demo2.yaml"
print(yaml.dump(data, default_flow_style=False))
# yaml.dump(data, "yaml_demo3.yaml")

f = open('yaml_demo3.yaml', 'w')
yaml.dump(data, f)

with open('yaml_demo4.yaml', 'w') as f:
    yaml.dump(data, f)

with open(new_file, 'w') as f:
    for line in yaml.dump(data, default_flow_style=False):
        f.write(line)