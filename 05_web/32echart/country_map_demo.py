# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from pyecharts import Map
value = [155, 10, 66, 78]
attr = ["福建", "山东", "北京", "上海"]
map = Map("全国地图示例",height=720,width=800)
map.add("", attr, value, maptype='china')
map.render("country_map.html")

