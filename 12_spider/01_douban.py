# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 豆瓣爬虫

网络数据采集是Python最擅长的领域之一。
"""


import random
import time

import requests
from bs4 import BeautifulSoup

for page in range(2):
    resp = requests.get(
        url=f'https://movie.douban.com/top250?start={25 * page}',
        headers={'User-Agent': 'BaiduSpider'}
    )
    soup = BeautifulSoup(resp.text, "lxml")
    for elem in soup.select('a > span.title:nth-child(1)'):
        print(elem.text)
    time.sleep(random.random() * 5)


