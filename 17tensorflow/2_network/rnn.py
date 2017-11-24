# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/11/23
# Brief: recurrent neural network

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

# mnist data
mnist = input_data.read_data_sets("../data/", one_hot=True)

"""
classify images using rnn, we consider every image row as a sequence of pixels.
"""

# training parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# network parameters
num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10

# tf graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# define weights
weights = {
    "out": tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    "out": tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):
    # unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    # define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def train():
    logits = RNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y
    ))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # evaluate model with test logits, dropout disabled
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # init
    init = tf.global_variables_initializer()

    # training
    with tf.Session() as sess:
        # run the init
        sess.run(init)

        for step in range(1, training_steps + 1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # run optimization op
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("step " + str(step) + ", minibatch loss= " + \
                      "{:.4f}".format(loss) + ", training accuracy= " + \
                      "{:.3f}".format(acc))
        print("optimization finished.")

        # calculate accuracy for 128 mnist test images
        test_len = 128
        test_data = mnist.test.images[:test_len].reshpae((-1, timesteps, num_input))
        test_label = mnist.test.labels[:test_len]
        print("testing accuracy:", \
              sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
train()