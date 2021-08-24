# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

def test_grad():
    import tensorflow as tf

    x = tf.Variable(initial_value=4.)
    with tf.GradientTape() as tape:  # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
        y = tf.square(x)
    y_grad = tape.gradient(y, x)  # 计算y关于x的导数
    print([y, y_grad])  # 2*x = 2*4 = 8
    # y = 16, y_grad = 8

def test_linear():
    import tensorflow as tf

    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])


    class Linear(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(
                units=1,
                activation=None,
                kernel_initializer=tf.zeros_initializer(),
                bias_initializer=tf.zeros_initializer()
            )

        def call(self, input):
            output = self.dense(input)
            return output


    # 以下代码结构与前节类似
    model = Linear()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pred = model(X)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
            loss = tf.reduce_mean(tf.square(y_pred - y))
        grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    print(model.variables)

if __name__ == '__main__':
    test_grad()
    test_linear()

    import tensorflow_datasets as tfds
    import tensorflow as tf
    import tensorflow.keras.layers as layers

    import time
    import numpy as np
    import matplotlib.pyplot as plt

    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)

    train_examples, val_examples = examples['train'], examples['validation']