# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 深度网络学习加法运算

# input '100+100'
# output '200'

import numpy as np
from keras import layers
from keras.models import Sequential
from six.moves import range


class CharTable(object):
    """
    Give a set of chars:
    encode chars to a one hot integer representation
    decode the one hot integer representation to their char output
    decode a vector of probs to their char output
    """

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """
        One hot encode given string C.
        :param C:
        :param num_rows: number of rows in the returned one hot encoding.
        :return:
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


# parameters
TRAINING_SIZE = 50000
DIGITS = 3  # max output is 999+999=1998
INVERT = True

MAXLEN = DIGITS + 1 + DIGITS
chars = '0123456789+ '
ctable = CharTable(chars)

questions = []
expected = []
seen = set()
print('make data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                            for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    # skip questions have seen
    # skip any such as 'a+b=b+a'
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # pad the data with spaces such that it is always maxlen
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # answers: max size of digits+1
    ans += ' ' * (DIGITS + 1 - len(ans))
    if INVERT:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('TOTAL questions:', len(questions))

print('vector...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

# shuffle (x,y)
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# explicitly set apart 10% for valid
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('training data:')
print(x_train.shape)
print(y_train.shape)
print('val data:')
print(x_val.shape)
print(y_val.shape)

# RNN, replace by GRU or SimpleRNN
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

print('Build model...')
model = Sequential()
# encode the input sentence
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# same as the decoder RNN input, repeatedly provide the last hidden state of RNN for each time step
# repeat 'digits+1' times to the max length of output
model.add(layers.RepeatVector(DIGITS + 1))
for i in range(LAYERS):
    # return sequences of (num_samples, timesteps, output_dim)
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# apply a dense layer to the every temporal slice of an input.
model.add(layers.TimeDistributed(layers.Dense(len(chars))))
model.add(layers.Activation('softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# train
for iteration in range(1, 20):
    print()
    print('-')
    print('iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    # select 10 sample from the validation set at random to visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print('\033[92m' + '☑' + '\033[0m', end=' ')
        else:
            print('\033[91m' + '☒' + '\033[0m', end=' ')
        print(guess)
