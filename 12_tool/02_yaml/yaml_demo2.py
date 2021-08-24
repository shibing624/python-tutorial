# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import yaml


class Monster(yaml.YAMLObject):
    yaml_tag = u'!Monster'

    def __init__(self, name, hp, ac, attacks):
        self.name = name
        self.hp = hp
        self.ac = ac
        self.attacks = attacks

    def __repr__(self):
        return "%s(name=%r, hp=%r, ac=%r, attacks=%r)" % (
            self.__class__.__name__, self.name, self.hp, self.ac, self.attacks)


cls = Monster(name="Caaa", hp=[2, 3, 3], ac=1, attacks=['BITG', "dfss"])
s = yaml.dump(cls)
print(s)

with open("monster.yaml", 'w') as f:
    yaml.dump(cls, f)

