# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from pyecharts import WordCloud
name_list = ['Sam S Club', 'Macys', 'Amy Schumer', 'Jurassic World',
            'Charter Communications','Chick Fil A', 'Planet Fitness',
            'Pitch Perfect', 'Express','Home', 'Johnny Depp',
            'Lena Dunham', 'Lewis Hamilton', 'KXAN', 'Mary EllenMark',
            'Farrah Abraham','Rita Ora', 'Serena Williams',
            'NCAA baseball tournament','Point Break']
value_list = [10000, 6181, 4386, 4055, 2467, 2244,
            1898, 1484, 1112,965, 847, 582, 555,
            550, 462, 366, 360, 282, 273, 265]
wordcloud = WordCloud(width=800, height=500)
wordcloud.add("", name_list, value_list, word_size_range=[20, 100])
wordcloud.render("wordcloud.html")