# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import keras

inputs = keras.layers.Input(shape=(784,))

# get tensor
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(4, activation='relu')(x)
preds = keras.layers.Dense(10, activation='softmax')(x)

model = keras.models.Model(inputs=inputs, outputs=preds)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Generate dummy data
import numpy as np

data = np.random.random((1000, 784))
labels = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
model.fit(data, labels, epochs=10, batch_size=64)
# loss: 2.2957 - acc: 0.1200

# model.save('my_seq_model.h5')
# del model
json_string = model.to_json()
print(json_string)
del model
import pickle

with open('seq.pkl', 'wb')as f:
    pickle.dump(json_string, f)

x_test = np.random.random((1000, 784))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# load model by file
# model = keras.models.load_model('my_seq_model.h5')
j = ''
with open('seq.pkl','rb') as f:
    j = pickle.load(f)

model = keras.models.model_from_json(j)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
score = model.evaluate(x_test, y_test, batch_size=128)
print('total loss on test set:', score[0])
print('accuracy of test set:', score[1])
# loss: 2.2999 - acc: 0.1140