# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: This example demonstrates the use of fasttext for text classification
# Bi-gram : 0.9056 test accuracy after 5 epochs.

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb


def create_ngram_set(input_list, ngram_value=2):
    """
    Create a set of n-grams
    :param input_list: [1, 2, 3, 4, 9]
    :param ngram_value: 2
    :return: {(1, 2),(2, 3),(3, 4),(4, 9)}
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list by appending n-grams values
    :param sequences:
    :param token_indice:
    :param ngram_range:
    :return:
    """
    new_seq = []
    for input in sequences:
        new_list = input[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_seq.append(new_list)
    return new_seq


ngram_range = 2
max_features = 20000
max_len = 400
batch_size = 32
embedding_dims = 50
epochs = 5

print('loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
train_mean_len = np.mean(list(map(len, x_train)), dtype=int)
test_mean_len = np.mean(list(map(len, x_test)), dtype=int)
print(len(x_train), 'train seq')
print(len(x_test), 'test seq')
print('Average train sequence length: {}'.format(train_mean_len))
print('Average test sequence length: {}'.format(test_mean_len))

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # n-gram set from train data
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            ng_set = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(ng_set)
    # add to ngram
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    max_features = np.max(list(indice_token.keys())) + 1
    # augment x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)

    train_mean_len = np.mean(list(map(len, x_train)), dtype=int)
    test_mean_len = np.mean(list(map(len, x_test)), dtype=int)
    print('Average train sequence length: {}'.format(train_mean_len))
    print('Average test sequence length: {}'.format(test_mean_len))

print('pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('build model...')
model = Sequential()

# embed layer by maps vocab index into emb dimensions
model.add(Embedding(max_features, embedding_dims, input_length=max_len))
# pooling the embedding
model.add(GlobalAveragePooling1D())
# squash with sigmoid into a single unit
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
