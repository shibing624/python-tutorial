# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import tensorflow as tf

x = tf.constant(1, tf.float32)
y = tf.nn.relu(x)
dy = tf.gradients(y, x)
ddy = tf.gradients(dy, x)
with tf.Session() as sess:
    print(sess.run(y))
    print(sess.run(dy))
    print(sess.run(ddy))
