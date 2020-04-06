# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_util
# 1.tf.function装饰器¶
# 当使用tf.function注释函数时，可以像调用任何其他函数一样调用它。 它将被编译成图，这意味着可以获得更快执行，更好地在GPU或TPU上运行或导出到SavedModel。
@tf.function
def simple_nn_layer(x, y):
    return tf.nn.softmax(tf.matmul(x, y))


x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

a = simple_nn_layer(x, y)
print(a)