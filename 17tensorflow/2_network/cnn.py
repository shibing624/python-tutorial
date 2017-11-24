# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/11/23
# Brief: convolutional_network

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/", one_hot=False)

# training params
learning_rate = 0.001
num_steps = 20  # 00
batch_size = 128

# network params
num_input = 784  # (28*28)
num_classes = 10
dropout = 0.5  # Dropout,probability to keep units


# network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # variables
    with tf.variable_scope("convnet", reuse=reuse):
        x = x_dict["images"]
        # tensor input is 4D: [batch size, height, width, channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # convolution layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # max pooling with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # convolution layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # max pooling
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # fully connected layer
        fc1 = tf.layers.dense(fc1, 1024)
        # apply dropout (only in test)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
    return out


# define the model function ( following TF estimator template)
def model_fn(features, labels, mode):
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)

    # prediction
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)
    ))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels, predictions=pred_classes)

    # TF estimators requires to return a estimatorSpec
    estimator_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op}
    )
    return estimator_specs


def train():
    # build the estimator
    model = tf.estimator.Estimator(model_fn=model_fn)
    # define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images}, y=mnist.train.labels,
        batch_size=batch_size, num_epochs=None, shuffle=True
    )
    # train the model
    model.train(input_fn, steps=num_steps)

    # evaluate the model
    # define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images}, y=mnist.train.labels,
        batch_size=batch_size, shuffle=False
    )
    # use the estimator 'evaluate' method
    e = model.evaluate(input_fn)

    print("testing accuracy:", e['accuracy'])


train()
