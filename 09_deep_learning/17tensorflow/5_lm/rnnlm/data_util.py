# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/11/29
# Brief: 
#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import os
import numpy as np


def _read_words(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '<eos>').split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[x] for x in data if x in word_to_id]

def to_words(sentence, words):
    return list(map(lambda x: words[x], sentence))

def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')

    words, word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    return train_data, valid_data, test_data, words, word_to_id

def ptb_producer(raw_data, batch_size=64, num_steps=20, stride=3):
    data_len = len(raw_data)

    sentences = []
    next_words = []
    for i in range(0, data_len - num_steps, stride):
        sentences.append(raw_data[i:(i + num_steps)])
        next_words.append(raw_data[i + num_steps])

    sentences = np.array(sentences)
    next_words = np.array(next_words)

    batch_len = len(sentences) // batch_size
    x = np.reshape(sentences[:(batch_len * batch_size)], \
        [batch_len, batch_size, -1])

    y = np.reshape(next_words[:(batch_len * batch_size)], \
        [batch_len, batch_size])

    return x, y


def main():
    train_data, valid_data, test_data, words, word_to_id = \
        ptb_raw_data('simple-examples/data')

    x_train, y_train = ptb_producer(train_data)

    print(x_train.shape)

    print(to_words(x_train[100, 3], words))

    print(words[np.argmax(y_train[100, 3])])

if __name__ == '__main__':
    main()