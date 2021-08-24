# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/16
# Brief: 
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=10))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=10))
x = tf.constant([[0.7, 0.9]])

# forward
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# session
sess = tf.Session()
sess.run(w1.initializer)
sess.run(w2.initializer)
print("w1", sess.run(w1))
print("w2", sess.run(w2))
print("x", sess.run(x))
print(sess.run(y))
sess.close()

# placeholder
x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
