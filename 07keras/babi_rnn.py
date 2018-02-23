# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

import os
import re
import tarfile
from functools import reduce

import keras
import numpy as np
from keras.models import Model
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


def read_lines(path):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if line:
                lines.append(line)
    return lines


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
    data = [(flatten(story), q, a) for story, q, a in data if not max_len or len(flatten(story)) < max_len]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    idx_story = []
    idx_query = []
    idx_answer = []
    for story, query, answer in data:
        s = [word_idx[w] for w in story]
        q = [word_idx[w] for w in query]
        a = np.zeros(len(word_idx) + 1)
        a[word_idx[answer]] = 1
        idx_story.append(s)
        idx_query.append(q)
        idx_answer.append(a)
    return pad_sequences(idx_story, maxlen=story_maxlen), pad_sequences(idx_query, maxlen=query_maxlen), np.array(idx_answer)


RNN = keras.layers.recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCH = 40
print("RNN,Embed,Sent,Query={},{},{},{}".format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))

challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
pwd_path = os.path.abspath(os.path.dirname(__file__))
print('pwd_path:', pwd_path)
path = os.path.join(pwd_path, '../data/babi_tasks_1-20_v1-2.tar.gz')
print('path:', path)
with tarfile.open(path) as tar:
    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, a in train + test:
    vocab |= set(story + q + [a])
vocab = sorted(vocab)

vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

idx_story, idx_query, idx_answer = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
test_idx_story, test_idx_query, test_idx_answer = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)
print('vocab:', vocab)
print('idx_story.shape:', idx_story.shape)
print('idx_query.shape:', idx_query.shape)
print('idx_answer.shape:', idx_answer.shape)
print('story max len:', story_maxlen)
print('query max len:', query_maxlen)

print('build model...')

sentence = keras.layers.Input(shape=(story_maxlen,), dtype='int32')
encoded_sentence = keras.layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
encoded_sentence = keras.layers.Dropout(0.3)(encoded_sentence)

question = keras.layers.Input(shape=(query_maxlen,), dtype='int32')
encoded_question = keras.layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
encoded_question = keras.layers.Dropout(0.3)(encoded_question)
encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
encoded_question = keras.layers.RepeatVector(story_maxlen)(encoded_question)

merged = keras.layers.add([encoded_sentence, encoded_question])
merged = RNN(EMBED_HIDDEN_SIZE)(merged)
merged = keras.layers.Dropout(0.3)(merged)
preds = keras.layers.Dense(vocab_size, activation='softmax')(merged)

model = Model([sentence, question], preds)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('training')
model.fit([idx_story, idx_query], idx_answer, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=0.05)
loss, acc = model.evaluate([test_idx_story, test_idx_query], test_idx_answer, batch_size=BATCH_SIZE)
print('Test loss / test accuracy= {:.4f} / {:.4f}'.format(loss, acc))
# loss: 1.6114 - acc: 0.3758 - val_loss: 1.6661 - val_acc: 0.3800
# Test loss / test accuracy= 1.6762 / 0.3050