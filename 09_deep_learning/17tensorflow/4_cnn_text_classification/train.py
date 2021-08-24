# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/17
# Brief: train data

import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import config
import data_helpers
from text_cnn import TextCNN
from util import to_categorical

# load data, label = (x_text, y)
x_text, y = data_helpers.load_data_labels(config.data_dir)

# build vocabulary
max_document_length = max([len(x.split(' ')) for x in x_text])
vocab_processr = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processr.fit_transform(x_text)))
y = to_categorical(y)

# randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# split train/test set
dev_sample_index = -1 * int(config.dev_sample_percentage * len(y))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("vocabulary size:{:d}".format(len(vocab_processr.vocabulary_)))
print("Train/dev split:.{:d}/{:d}".format(len(y_train), len(y_dev)))

# train
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=config.allow_soft_placement,
                                  log_device_placement=config.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processr.vocabulary_),
            embedding_size=config.embedding_dim,
            filter_sizes=list(map(int, config.filter_sizes.split(","))),
            num_filters=config.num_filters,
            l2_reg_lambda=config.l2_reg_lambda
        )

        # define training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_total = tf.summary.merge(grad_summaries)

        # output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "models"))
        print("writing to {}\n".format(out_dir))

        # summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # train summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_total])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # checkpoint directory, need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)

        # write vocabulary
        vocab_processr.save(os.path.join(out_dir, "vocab"))

        # init
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            :param x_batch:
            :param y_batch:
            :return:
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: config.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluate model on dev set
            :param x_batch:
            :param y_batch:
            :param write:
            :return:
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        # generate batches
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), config.batch_size,
                                          config.num_epochs)

        # training loop
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % config.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % config.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
