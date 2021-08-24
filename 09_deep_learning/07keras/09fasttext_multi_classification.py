# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: This example demonstrates the use of fasttext for text classification
# Bi-gram : 0.9056 test accuracy after 5 epochs.
import os

import keras
import numpy as np
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing import sequence


def get_corpus(data_dir):
    """
    Get the corpus data with retrieve
    :param data_dir:
    :return:
    """
    words = []
    labels = []
    for file_name in os.listdir(data_dir):
        with open(os.path.join(data_dir, file_name), mode='r', encoding='utf-8') as f:
            for line in f:
                # label in first sep
                parts = line.rstrip().split(',', 1)
                if parts and len(parts) > 1:
                    # keras categorical label start with 0
                    lbl = int(parts[0]) - 1
                    sent = parts[1]
                    sent_split = sent.split()
                    words.append(sent_split)
                    labels.append(lbl)
    return words, labels


def vectorize_words(words, word_idx):
    inputs = []
    for word in words:
        inputs.append([word_idx[w] for w in word])
    return inputs


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
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
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
num_classes = 3
max_features = 20000
max_len = 400
batch_size = 32
embedding_dims = 50
epochs = 10
SAVE_MODEL_PATH = 'fasttext_multi_classification_model.h5'
pwd_path = os.path.abspath(os.path.dirname(__file__))
print('pwd_path:', pwd_path)
train_data_dir = os.path.join(pwd_path, '../data/sogou_classifier_data/train')
test_data_dir = os.path.join(pwd_path, '../data/sogou_classifier_data/test')
print('data_dir path:', train_data_dir)

print('loading data...')
x_train, y_train = get_corpus(train_data_dir)
x_test, y_test = get_corpus(test_data_dir)
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

sent_maxlen = max(map(len, (x for x in x_train + x_test)))
print('-')
print('Sentence max length:', sent_maxlen, 'words')
print('Number of training data:', len(x_train))
print('Number of test data:', len(x_test))
print('-')
print('Here\'s what a "sentence" tuple looks like (label, sentence):')
print(y_train[0], x_train[0])
print('-')
print('Vectorizing the word sequences...')

print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

vocab = set()
for w in x_train + x_test:
    vocab |= set(w)
vocab = sorted(vocab)
vocab_size = len(vocab) + 1
print('Vocab size:', vocab_size, 'unique words')
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
ids_2_word = dict((value, key) for key, value in word_idx.items())

x_train = vectorize_words(x_train, word_idx)
x_test = vectorize_words(x_test, word_idx)

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # n-gram set from train data
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            ng_set = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(ng_set)
    # add to n-gram
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

print('build model...')
model = Sequential()

# embed layer by maps vocab index into emb dimensions
model.add(Embedding(max_features, embedding_dims, input_length=max_len))
# pooling the embedding
model.add(GlobalAveragePooling1D())
# output multi classification of num_classes
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
model.save(SAVE_MODEL_PATH)
print('save model:', SAVE_MODEL_PATH)
probs = model.predict(x_test, batch_size=batch_size)
assert len(probs) == len(y_test)
for label, prob in zip(y_test, probs):
    print('label_test_index:%s\tprob_index:%s\tprob:%s' % (label.argmax(), prob.argmax(), prob.max()))
