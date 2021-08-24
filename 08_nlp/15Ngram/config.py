# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/8/29
# Brief: 

train_data_path = '../data/ngram_wordseg/TrainData.txt'
test_data_path = '../data/ngram_wordseg/test.txt'
test_result_path = '../data/ngram_wordseg/result.txt'
test_gold_path = '../data/ngram_wordseg/gold.txt'

Punctuation = [u'、', u'”', u'“', u'。', u'（', u'）', u'：', u'《', u'》', u'；', u'！', u'，', u'、']

span = 16

Number = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'%', u'.']

English = [u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r',
           u's', u't', u'u', u'v', u'w', u'x', u'y', u'z',
           u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R',
           u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z']
