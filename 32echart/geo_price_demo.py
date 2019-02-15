# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from pyecharts import Geo

data = [("合肥", 11229), ("武汉", 20273), ("大庆", 5679)]

cities = ["合肥", "武汉", "大庆"]
prices = [11229, 20273, 5679]

geo = Geo(
    "全国主要城市房价",
    "data from ke.com",
    title_color="#fff",
    title_pos="center",
    width=1200,
    height=600,
    background_color="#404a59",
)
attr, value = geo.cast(data)
geo.add(
    "",
    cities,
    prices,
    visual_range=[5000, 22000],
    visual_text_color="#fff",
    symbol_size=15,
    is_visualmap=True,
)
geo.render("geo_price.html")
