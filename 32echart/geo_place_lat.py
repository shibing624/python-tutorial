# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from pyecharts.datasets.coordinates import get_coordinate,search_coordinates_by_keyword

coordinate = get_coordinate('武汉', region="中国")
print(coordinate)

ret = search_coordinates_by_keyword("东湖区")
print(ret)
