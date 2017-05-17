# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理
from Sentiment import SentimentIntensityAnalyzer
# from nltk.sentiment import SentimentIntensityAnalyzer

sentences = ["lilei is smart, handsome, and funny boy.",
             "lilei is not smart , handsome, nor good boy."]
analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))
