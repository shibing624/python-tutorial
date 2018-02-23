# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import keras
import numpy as np

tweet_a = keras.layers.Input(shape=(280, 256))
tweet_b = keras.layers.Input(shape=(280, 256))

shared_lstm = keras.layers.LSTM(64)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# concatenate the two vectors
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)
preds = keras.layers.Dense(1, activation='sigmoid')(merged_vector)

model = keras.models.Model(inputs=[tweet_a, tweet_b], outputs=preds)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

data_a = np.random.random((1000, 280, 256))
data_b = np.random.random((1000, 280, 256))
labels = np.random.randint(2, size=(1000, 1))
model.fit([data_a, data_b], labels, epochs=10, batch_size=64)
# loss: 0.5961 - acc: 0.7230
