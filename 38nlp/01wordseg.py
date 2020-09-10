# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import jieba

text = "一个傻子在北京, 哈尔滨工业大学迎来100年华诞，周华主持了会议。乐视赞助了会议"
print(jieba.lcut(text))

import jieba.posseg
print(jieba.posseg.lcut(text))

import pkuseg

seg = pkuseg.pkuseg()  # 以默认配置加载模型
text = seg.cut(text)  # 进行分词
print(text)
#
# seg = pkuseg.pkuseg(postag=True)  # 以默认配置加载模型
# text = seg.cut(text)  # 进行分词
# print(text)
