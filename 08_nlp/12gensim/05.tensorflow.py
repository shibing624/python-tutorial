# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""



import tensorflow as tf

x = tf.constant(1, tf.float32)
y = tf.nn.relu(x)
dy = tf.gradients(y, x)
ddy = tf.gradients(dy, x)
with tf.Session() as sess:
    print(sess.run(y))
    print(sess.run(dy))
    print(sess.run(ddy))
