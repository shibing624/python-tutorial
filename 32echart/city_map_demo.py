# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from pyecharts import Map

value = [20, 190, 253, 77, 65]
attr = ['汉阳区', '汉口区', '武昌区', '洪山区', '青山区']
map = Map("武汉地图示例", width=1200,
          height=600)
map.add("", attr, value, maptype='武汉',
        is_visualmap=True,
        visual_text_color='#000', is_label_show=True)
map.render("city_map.html")
