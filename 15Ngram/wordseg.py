# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/8/29
# Brief: 

import sys
import os
import re

StopWordtmp = ['。', '，', '！', '？', '：', '“', '”', '‘', '’', '（', '）', '【', '】', '｛', '｝', '-', '－', '～',
               '［', '］', '〔', '〕', '．', '＠', '￥', '•', '.']
# StopWordtmp = [' ', u'\u3000', u'\x30fb', u'\u3002', u'\uff0c', u'\uff01', u'\uff1f', u'\uff1a', u'\u201c', u'\u201d', u'\u2018', u'\u2019', u'\uff08', u'\uff09', u'\u3010', u'\u3011', u'\uff5b', u'\uff5d', u'-', u'\uff0d', u'\uff5e', u'\uff3b', u'\uff3d', u'\u3014', u'\u3015', u'\uff0e', u'\uff20', u'\uffe5', u'\u2022', u'.']

WordDict = {}
StopWord = []
StatisticDict = {}
span = 16


def InitStopword():
    for key in StopWordtmp:
        StopWord.append(key)


def InitDict(dict_file, encode="utf-8"):
    with open(dict_file, encoding=encode) as f:
        for line in f:
            line = line.strip()
            WordDict[line] = 1
    print("Dictionary init success.count:[" + str(len(WordDict)) + "]")


def InitStatisticDict(statistic_dict_file, encode="utf-8"):
    StatisticDict['<B>'] = {}
    with open(statistic_dict_file, encoding=encode) as f:
        for line in f:
            chunk = line.strip().split('  ')
            if chunk[0] != '':
                if chunk[0] not in StatisticDict['<B>']:
                    StatisticDict['<B>'][chunk[0]] = 1
                else:
                    StatisticDict['<B>'][chunk[0]] += 1

            for i in range(len(chunk) - 1):
                if chunk[i] not in StatisticDict and chunk[i] != '':
                    StatisticDict[chunk[i]] = {}
                if chunk[i] != '':
                    if chunk[i + 1] not in StatisticDict[chunk[i]]:
                        StatisticDict[chunk[i]][chunk[i + 1]] = 1
                    else:
                        StatisticDict[chunk[i]][chunk[i + 1]] += 1

            if chunk[-1] not in StatisticDict and chunk[-1] != '':
                StatisticDict[chunk[-1]] = {}
            if chunk[-1] != '':
                if '<E>' not in StatisticDict[chunk[-1]]:
                    StatisticDict[chunk[-1]]['<E>'] = 1
                else:
                    StatisticDict[chunk[-1]]['<E>'] += 1


def WordSeg(input_file, output_file, encode="utf-8"):
    dict_size = 0
    for key in StatisticDict:
        for keys in StatisticDict[key]:
            dict_size += StatisticDict[key][keys]

    with open(input_file, 'r', encoding=encode) as rf:
        for line in rf:
            line = line.strip()
            sent_list = []
            new_sent_list = []
            temp_word = ''
            for i in range(len(line)):
                if line[i] in StopWord:
                    sent_list.append(temp_word)
                    sent_list.append(line[i])
                    temp_word = ''
                else:
                    temp_word += line[i]
                    if i == len(line) - 1:
                        sent_list.append(temp_word)

            # N-gram
            for key in sent_list:
                if key in StopWord:
                    new_sent_list.append(key)
                else:
                    PreTempList = PreSentSeg(key, span)
                    PostTempList = PostSentSeg(key, span)
                    temp_pre = P(PreTempList, dict_size)
                    temp_post = P(PostTempList, dict_size)
                    temp_list = []
                    if temp_pre > temp_post:
                        temp_list = PreTempList
                    else:
                        temp_list = PostTempList
                    for i in temp_list:
                        new_sent_list.append(i)
            with open(output_file, encoding=encode, mode='a') as wf:
                write_line = ''
                for key in new_sent_list:
                    write_line = write_line + key + "\t"
                write_line = write_line.strip('\t')
                wf.write(write_line + "\n")


def P(temp_list, dict_size):
    rev = 1
    if len(temp_list) < 1:
        return 0
    rev *= Pword(temp_list[0], '<B>', dict_size)
    rev *= Pword('<E>', temp_list[-1], dict_size)
    for i in range(len(temp_list) - 1):
        rev *= Pword(temp_list[i + 1], temp_list[i], dict_size)
    return rev


def Pword(word1, word2, dict_size):
    div_up = 0
    div_down = 0
    if word2 in StatisticDict:
        for key in StatisticDict[word2]:
            div_down += StatisticDict[word2][key]
            if key == word1:
                div_up = StatisticDict[word2][key]
    return (div_up + 1) / (div_down + dict_size+1)


def PreSentSeg(sent, span):
    post = span
    if len(sent) < span:
        post = len(sent)
    cur = 0
    revlist = []
    while 1:
        if cur >= len(sent):
            break
        if (sent[cur:post] in WordDict) or (cur + 1 == post):
            if sent[cur:post] != '':
                revlist.append(sent[cur:post])
            cur = post
            post = post + span
            if post > len(sent):
                post = len(sent)
        else:
            post -= 1
    return revlist


def PostSentSeg(sent, span):
    cur = len(sent)
    pre = cur - span
    if pre < 0:
        pre = 0
    revlist = []
    while 1:
        if cur <= 0:
            break
        if (sent[pre:cur] in WordDict) or (cur - 1 == pre):
            if sent[pre:cur] != '':
                revlist.append(sent[pre:cur])
            cur = pre
            pre = pre - span
            if pre < 0:
                pre = 0
        else:
            pre += 1
    return revlist[::-1]


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: python wordseg.py Dicfile inputfile outputfile")
    dict_file = sys.argv[1]
    statistic_dict_file = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    InitDict(dict_file)
    InitStatisticDict(statistic_dict_file)
    InitStopword()
    WordSeg(input_file, output_file)
