# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: memory network
# - Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
#   "End-To-End Memory Networks",
#   http://arxiv.org/abs/1503.08895
#
# Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
# Time per epoch: 3s on CPU (core i7).

import os
import tarfile
import keras
from keras.layers import Input, Activation, Dense, Dropout, add, dot, Permute, concatenate
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model

from babi.util import get_stories
from babi.util import memnn_vectorize_stories


def network_conf(story_maxlen,
                 query_maxlen,
                 vocab_size,
                 hidden_layer_size=64):
    input_sequence = Input((story_maxlen,))
    question = Input((query_maxlen,))

    # encoders
    # embed the input sequence into a sequence of vectors
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=hidden_layer_size))
    input_encoder_m.add(Dropout(0.3))

    # embed the input sequence into a sequence of query maxlen size vectors
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_maxlen))
    input_encoder_c.add(Dropout(0.3))

    # embed the question into a sequence of vectors
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size, output_dim=hidden_layer_size, input_length=query_maxlen))
    question_encoder.add(Dropout(0.3))

    # encode input sequence and questions
    input_encoder_m = input_encoder_m(input_sequence)
    input_encoder_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)

    # compute the similarity between first input and question
    match = dot([input_encoder_m, question_encoded], axes=(2, 2))
    match = Activation('softmax')(match)

    # add the second input to match matrix
    response = add([match, input_encoder_c])  # sample, story, query
    response = Permute((2, 1))(response)  # sample, query, story

    # concatenate the match matrix with question
    answer = concatenate([response, question_encoded])
    # RNN
    answer = LSTM(32)(answer)
    answer = Dropout(0.3)(answer)
    answer = Dense(vocab_size)(answer)
    # output probability distribution over vocabulary
    answer = Activation('softmax')(answer)

    # model
    model = Model([input_sequence, question], answer)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def reader(train_stories, test_stories):
    vocab = set()
    for story, q, answer in train_stories + test_stories:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', story_maxlen, 'words')
    print('Query max length:', query_maxlen, 'words')
    print('Number of training stories:', len(train_stories))
    print('Number of test stories:', len(test_stories))
    print('-')
    print('Here\'s what a "story" tuple looks like (input, query, answer):')
    print(train_stories[0])
    print('-')
    print('Vectorizing the word sequences...')

    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    ids_2_word = dict((value, key) for key, value in word_idx.items())

    return story_maxlen, query_maxlen, vocab_size, word_idx, ids_2_word


def train():
    model = network_conf(story_maxlen, query_maxlen, vocab_size)
    model.fit([inputs_train, queries_train], answers_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCH,
              validation_data=([inputs_test, queries_test], answers_test))
    model.save(SAVE_MODEL_PATH)
    print('save model:', SAVE_MODEL_PATH)
    probs = model.predict([inputs_test, queries_test], batch_size=BATCH_SIZE)
    assert len(probs) == len(answers_test)
    for answer, prob in zip(answers_test, probs):
        print('answer_test_index:%s\tprob_index:%s\tprob:%s' % (answer, prob.argmax(), prob.max()))


def predict():
    if os.path.exists(SAVE_MODEL_PATH):
        model = keras.models.load_model(SAVE_MODEL_PATH)
    else:
        return

    loss, acc = model.evaluate([inputs_test, queries_test], answers_test, batch_size=BATCH_SIZE)
    print('Test loss / test accuracy= {:.4f} / {:.4f}'.format(loss, acc))
    probs = model.predict([inputs_test, queries_test], batch_size=BATCH_SIZE)
    assert len(probs) == len(answers_test)
    for story, prob in zip(test_stories, probs):
        print('story:%s\tprob:%s' % (story, ids_2_word[prob.argmax()]))
    for answer, prob in zip(answers_test, probs):
        print('answer_test_index:%s\tprob_index:%s\tprob:%s' % (answer, prob.argmax(), prob.max()))


if __name__ == '__main__':
    BATCH_SIZE = 32
    EPOCH = 100
    SAVE_MODEL_PATH = 'babi_memnn_model.h5'
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    print('pwd_path:', pwd_path)
    path = os.path.join(pwd_path, '../../data/babi_tasks_1-20_v1-2.tar.gz')
    print('path:', path)
    challenges = {
        # QA1 with 10,000 samples
        'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
        # QA2 with 10,000 samples
        'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    }
    challenge_type = 'single_supporting_fact_10k'
    challenge = challenges[challenge_type]
    print('Extracting stories for the challenge:', challenge_type)
    with tarfile.open(path) as tar:
        train_stories = get_stories(tar.extractfile(challenge.format('train')))
        test_stories = get_stories(tar.extractfile(challenge.format('test')))

    story_maxlen, query_maxlen, vocab_size, word_idx, ids_2_word = reader(train_stories, test_stories)
    inputs_train, queries_train, answers_train = memnn_vectorize_stories(train_stories, word_idx, story_maxlen,
                                                                         query_maxlen)
    inputs_test, queries_test, answers_test = memnn_vectorize_stories(test_stories, word_idx, story_maxlen,
                                                                      query_maxlen)
    print('-')
    print('inputs: integer tensor of shape (samples, max_length)')
    print('inputs_train shape:', inputs_train.shape)
    print('inputs_test shape:', inputs_test.shape)
    print('-')
    print('queries: integer tensor of shape (samples, max_length)')
    print('queries_train shape:', queries_train.shape)
    print('queries_test shape:', queries_test.shape)
    print('-')
    print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
    print('answers_train shape:', answers_train.shape)
    print('answers_test shape:', answers_test.shape)
    train()
    # predict()
