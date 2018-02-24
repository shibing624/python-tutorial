# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

import os
import tarfile

import keras
from keras.models import Model

from babi.util import get_stories
from babi.util import vectorize_stories

RNN = keras.layers.recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCH = 2
save_model_path = 'babi_rnn_model.h5'
print("RNN,Embed,Sent,Query={},{},{},{}".format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))

challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
pwd_path = os.path.abspath(os.path.dirname(__file__))
print('pwd_path:', pwd_path)
path = os.path.join(pwd_path, '../../data/babi_tasks_1-20_v1-2.tar.gz')
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


def train():
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

    model.save(save_model_path)
    print('save model:', save_model_path)


if __name__ == '__main__':
    train()
