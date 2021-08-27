# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Embedding(128, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=2)
score = model.evaluate(x_test, y_test, batch_size=128)
print('total loss on test set:', score[0])
print('accuracy of test set:', score[1])

# from keras.utils import plot_model
# plot_model(model, to_file='lstm_model.png')

model.save('lstm_model.h5')
del model
# load model by file
model = keras.models.load_model('lstm_model.h5')
score = model.evaluate(x_test, y_test, batch_size=128)
print('total loss on test set:', score[0])
print('accuracy of test set:', score[1])

x_test = np.random.random((200, 20))
y_test = np.random.randint(2, size=(200, 1))
score = model.evaluate(x_test, y_test, batch_size=128)
print('total loss on test set:', score[0])
print('accuracy of test set:', score[1])