# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

# 构建一个简单的模型并训练
from __future__ import absolute_import, division, print_function
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1)

predictions = model.predict(x_test)

# 保持为全模型
model.save('the_save_model.h5')
new_model = keras.models.load_model('the_save_model.h5')
new_prediction = new_model.predict(x_test)
print('predictions',predictions[:10], 'new_prediction',new_prediction[:10])
np.testing.assert_allclose(predictions, new_prediction, atol=1e-6) # 预测结果一样

# 1.2 保持为SavedModel文件
keras.experimental.export_saved_model(model, 'saved_model')
new_model = keras.experimental.load_from_saved_model('saved_model')
new_prediction = new_model.predict(x_test)
print('predictions',predictions[:10], 'new_prediction',new_prediction[:10])
np.testing.assert_allclose(predictions, new_prediction, atol=1e-6) # 预测结果一样

