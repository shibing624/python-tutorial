# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/16
# Brief: 
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable('v', [1], initializer=tf.zeros_initializer())

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", [1], initializer=tf.ones_initializer())

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')
c = a + b
print(c)
sess = tf.InteractiveSession()
print(c.eval())
sess.close()

with tf.Session() as sess:
    print("a+b:")
    print(sess.run(c))

# configProto
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
print("configproto:")
print(sess1.run(c))
print(sess2.run(c))
