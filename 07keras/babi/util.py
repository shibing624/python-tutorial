# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 

import re
from functools import reduce

import numpy as np
from keras.preprocessing.sequence import pad_sequences


def tokenize(sentence):
    """
    English segment
    :param sentence:
    :return:
    """
    return [x.strip() for x in re.split('(\W+)?', sentence) if x.strip()]


def parse_stroes(lines, only_supporting=False):
    """
    Parse stories by bAbi task format
    :param lines:
    :param only_supporting:
    :return:
    """
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        id, line = line.split(' ', 1)
        id = int(id)
        if id == 1:
            story = []
        if '\t' in line:
            q, a, support = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # only select the related substory
                support = map(int, support.split(' '))
                substory = [story[i - 1] for i in support]
            else:
                # get all the substory
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_len=None):
    """
    Get the stories with retrieve, and convert the sentences into a single story
    :param f:
    :param only_supporting:
    :param max_len:
    :return:
    """
    data = parse_stroes(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, a) for story, q, a in data \
            if not max_len or len(flatten(story)) < max_len]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answer_ids = np.zeros(len(word_idx) + 1)
        answer_ids[word_idx[answer]] = 1
        answers.append(answer_ids)
    return pad_sequences(inputs, maxlen=story_maxlen), \
           pad_sequences(queries, maxlen=query_maxlen), \
           np.array(answers)


def memnn_vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return pad_sequences(inputs, maxlen=story_maxlen),\
            pad_sequences(queries, maxlen=query_maxlen),\
            np.array(answers)
