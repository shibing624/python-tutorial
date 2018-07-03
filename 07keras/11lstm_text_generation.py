# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: generate text from hand writings

import os
import random
import sys

import numpy as np
from keras.callbacks import LambdaCallback
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

SAVE_MODEL_PATH = 'text_generation_model.h5'
pwd_path = os.path.abspath(os.path.dirname(__file__))
print('pwd_path:', pwd_path)
data_path = os.path.join(pwd_path, '../data/nietzsche.txt')
print('data path:', data_path)


def get_corpus(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return text


text = get_corpus(data_path)
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut sequences of max len chars
maxlen = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])
print('num sentences:', len(sentences))

print('vector...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build LSTM model
print('build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(optimizer=RMSprop(lr=0.01), loss='categorical_crossentropy')
model.summary()

print("*"*40)
print(model.summary())
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)


def on_epoch_end(epoch):
    # print generated text
    print('\n--- Generating text each epoch: %d' % epoch)
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('--- diversity:', diversity)
        generated = ''
        sentence = text[start_index:start_index + maxlen]
        generated += sentences
        print('--- generating with:', sentence)
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zero((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x, y,
          batch_size=128,
          epochs=60,
          callbacks=[print_callback])
model.save(SAVE_MODEL_PATH)
