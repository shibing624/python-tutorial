# -*- coding: utf-8 -*-

"""
@author: XuMing <shibing624@126.com>
@summary:
"""
import matplotlib.pyplot as plt
import pickle
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import jieba

text = open('../data/tianlongbabu.txt', encoding='utf8').read()
text = ' '.join(jieba.lcut(text))
with open('../data/stopword.txt', encoding='utf-8') as f:
    for line in f:
        STOPWORDS.add(line.strip())
print("stopwrod size:", len(STOPWORDS))
backgroud_Image = plt.imread('../data/cloud/girl.jpg')
wc = WordCloud(background_color='white',  # 设置背景颜色
               # mask=backgroud_Image,  # 设置背景图片
               # max_words=2000,  # 设置最大现实的字数
               stopwords=STOPWORDS,  # 设置停用词
               font_path='/System/Library/Fonts/STHeiti Light.ttc',
               # font_path = 'C:/Users/Windows/fonts/msyh.ttf',# 设置字体格式，如不设置显示不了中文
               )
wc.generate(text)
plt.imshow(wc)
plt.axis('off')
plt.show()
