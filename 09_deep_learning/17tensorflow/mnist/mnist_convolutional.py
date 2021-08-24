# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/10
# Brief:

import gzip
import os
import tempfile
import time
import argparse
import numpy as np
import sys
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"
WORK_DIRECTORY = '../data'
IMAGE_SIZE = 28  # 图像尺寸：28 * 28
NUM_CHANNELS = 1  # 黑白图像
PIXEL_DEPTH = 255  # 像素值0——255
NUM_LABELS = 10  # 10个类别，0-9的10个数字
VALIDATION_SIZE = 500  # 验证集大小
SEED = 66421  # 随机数种子
BATCH_SIZE = 64  # 批处理大小为64
NUM_EPOCHS = 10  # 数据全集过10遍网络
EVAL_BATCH_SIZE = 64  # 验证集批处理也是64
EVAL_FREQUENCY = 100  # 每训练100个批处理，做一次评估

FLAGS = None


def data_type():
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32


def extract_data(filename, num_images):
    """
    转换图像数据为4D张量[image index, y, x, channels]
    值从[0, 255]归一化[-0.5, 0.5]
    :param filename:
    :param num_images:
    :return:
    """
    print("Extracting", filename)
    with gzip.open(filename)as b:
        b.read(16)
        buf = b.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data


def extract_labels(filename, num_images):
    """
    提取图像的数字标签值（0-9）
    :param filename:
    :param num_images:
    :return:
    """
    print("Extracting", filename)
    with gzip.open(filename)as b:
        b.read(8)
        buf = b.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def fake_data(num_images):
    """
    生成与MNIST维度匹配的数据集
    :param num_images:
    :return:
    """
    data = np.ndarray(shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
    labels = np.zeros(shape=(num_images,), dtype=np.int64)
    for image in xrange(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image] = label
    return data, labels


def error_rate(predictions, labels):
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])


def maybe_download(filename):
    """
    下载数据集，如果已经下载就不重复下载
    :param filename:
    :return:
    """
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, i = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print("download success.", filename, size, 'bytes.')
    return filepath


def main(_):
    if FLAGS.self_test:
        print("Running test.")
        train_data, train_labels = fake_data(256)
        validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
        test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
        num_epochs = 1
    else:
        # Get the data.
        train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
        train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
        test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
        test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

        # Extract it into numpy arrays.
        train_data = extract_data(train_data_filename, 60000)
        train_labels = extract_labels(train_labels_filename, 60000)
        test_data = extract_data(test_data_filename, 10000)
        test_labels = extract_labels(test_labels_filename, 10000)

        # generate a validation set
        validation_data = train_data[:VALIDATION_SIZE, ...]
        validation_labels = train_labels[:VALIDATION_SIZE]
        train_data = train_data[VALIDATION_SIZE:, ...]
        train_labels = train_data[VALIDATION_SIZE:]
        num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]

    # training samples
    train_data_node = tf.placeholder(data_type(), shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(data_type(), shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    # train weights
    conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                                    stddev=0.1, seed=SEED, dtype=data_type()))
    conv2_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
    conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=data_type()))
    conv1_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
    fc1_weights = tf.Variable(
        tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], stddev=0.1, seed=SEED, dtype=data_type())
    )
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED, dtype=data_type()))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))

    # 转换训练数据为图模型格式
    def model(data, train=False):
        """
        定义模型
        :param data:
        :param train:
        :return: [image index, y, x, depth]
        """
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        # 偏执并将线性转为非线性
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # max pooling
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 把feature map 映射为二维矩阵，以将其传给全连接层
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # 全连接层
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # 训练中加入50% dropout
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # computation: logits + cross-entrogy loss
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_data_node, logits=logits))

    # 全连接层 L2 正则化
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # 添加正则结果入损失值
    loss += 5e-4 * regularizers

    # 优化器，设置一个变量，每批增加一次，并控制学习速率的衰减。
    batch = tf.Variable(0, dtype=data_type())
    # 每轮衰减，起始是0.01
    learning_rate = tf.train.exponential_decay(
        0.01,  # 初始学习率
        batch * BATCH_SIZE,  # index
        train_size,  # 衰减步长
        0.95,  # 衰减率
        staircase=True
    )
    # 使用momentum梯度下降优化器
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

    # 当前训练的 minibatch 的准确率
    train_prediction = tf.nn.softmax(logits)

    # 测试准确率
    eval_prediction = tf.nn.softmax(model(eval_data))

    # 评估 batch 数据集的准确率
    def eval_in_batches(data, sess):
        """
        Get all predictions for a dataset by running it in small batches.
        :param data:
        :param sess:
        :return:
        """
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(eval_prediction, feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(eval_prediction, feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    # 新建session，训练数据
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("initialized.")
        # 循环
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            # 计算offset
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
            # 更新权重weights
            sess.run(optimizer, feed_dict=feed_dict)
            if step % EVAL_FREQUENCY == 0:
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction], feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print("step %d (epoch %.2f), %.1f ms" % (
                    step, float(step) * BATCH_SIZE / train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % error_rate(
                    eval_in_batches(validation_data, sess), validation_labels))
                sys.stdout.flush()
            # print result
            test_err = error_rate(eval_in_batches(test_data, sess), test_labels)
            print("test error: %.1f%%" % test_err)
            if FLAGS.self_test:
                print("test error", test_err)
                assert test_err == 0.0, 'expected 0.0 test_error, got %.2f' % (test_err,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_fp16',
        default=False,
        help='Use half floats instead of full floats if True.',
        action='store_true')
    parser.add_argument('--self_test', default=False, action='store_true', help='True if test')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
