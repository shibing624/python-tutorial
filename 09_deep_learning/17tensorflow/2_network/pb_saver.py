# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/16
# Brief: 
import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
v3 = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print("v3:", sess.run(v3))
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    with tf.gfile.GFile("../data/models/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())

# 加载pb文件
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "../data/models/combined_model.pb"

    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))
