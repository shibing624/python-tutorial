# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/16
# Brief: 
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
v3 = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "../data/models/model.ckpt")

# 加载两个变量和模型
with tf.Session() as sess:
    saver.restore(sess, "../data/models/model.ckpt")
    print(sess.run(v3))

# 加载持久化图
saver_graph = tf.train.import_meta_graph("../data/models/model.ckpt.meta")
with tf.Session() as sess:
    saver_graph.restore(sess, "../data/models/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

# 变量重命名
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
saver = tf.train.Saver({"v1": v1, "v2": v2})
