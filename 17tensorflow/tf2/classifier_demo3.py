# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import matplotlib.pyplot as plt
import tensorflow.keras as keras

(train_data, train_label), (test_data, test_label) = keras.datasets.fashion_mnist.load_data()
print(train_data.shape)
print(train_label.shape)


model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'],
              )
his = model.fit(train_data, train_label, epochs=20, validation_data=(test_data, test_label))
print(his.history.keys())

plt.plot(his.epoch, his.history.get('loss'), label='loss')
plt.plot(his.epoch, his.history.get('val_loss'), label='val_loss')
plt.legend()
plt.savefig('c5.png')
plt.close()

plt.plot(his.epoch, his.history.get('acc'), label='acc')
plt.plot(his.epoch, his.history.get('val_acc'), label='val_acc')
plt.legend()
plt.savefig('c6.png')
plt.close()