# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/11/29
# Brief: 

from data_util import *
import tensorflow as tf

class LMConfig(object):
    """Configuration of language model"""
    batch_size = 64
    num_steps = 20
    stride = 3

    embedding_dim = 64
    hidden_dim = 128
    num_layers = 2
    rnn_model = 'gru'

    learning_rate = 0.05
    dropout = 0.2


class PTBInput(object):
    def __init__(self, config, data):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.vocab_size = config.vocab_size

        self.input_data, self.targets = ptb_producer(data,
            self.batch_size, self.num_steps)

        self.batch_len = self.input_data.shape[0]
        self.cur_batch = 0

    def next_batch(self):
        x = self.input_data[self.cur_batch]
        y = self.targets[self.cur_batch]

        y_ = np.zeros((y.shape[0], self.vocab_size), dtype=np.bool)
        for i in range(y.shape[0]):
            y_[i][y[i]] = 1

        self.cur_batch = (self.cur_batch +1) % self.batch_len

        return x, y_


class PTBModel(object):
    def __init__(self, config, is_training=True):

        self.num_steps = config.num_steps
        self.vocab_size = config.vocab_size

        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.rnn_model = config.rnn_model

        self.learning_rate = config.learning_rate
        self.dropout = config.dropout

        self.placeholders()
        self.rnn()
        self.cost()
        self.optimize()
        self.error()


    def placeholders(self):
        self._inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self._targets = tf.placeholder(tf.int32, [None, self.vocab_size])


    def input_embedding(self):
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [self.vocab_size,
                    self.embedding_dim], dtype=tf.float32)
            _inputs = tf.nn.embedding_lookup(embedding, self._inputs)

        return _inputs


    def rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_dim,
                state_is_tuple=True)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.hidden_dim)

        def dropout_cell():
            if (self.rnn_model == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell,
                output_keep_prob=self.dropout)

        cells = [dropout_cell() for _ in range(self.num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        _inputs = self.input_embedding()
        _outputs, _ = tf.nn.dynamic_rnn(cell=cell,
            inputs=_inputs, dtype=tf.float32)

        last = _outputs[:, -1, :]
        logits = tf.layers.dense(inputs=last, units=self.vocab_size)
        prediction = tf.nn.softmax(logits)

        self._logits = logits
        self._pred = prediction

    def cost(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self._logits, labels=self._targets)
        cost = tf.reduce_mean(cross_entropy)
        self.cost = cost

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optim = optimizer.minimize(self.cost)

    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self._targets, 1), tf.argmax(self._pred, 1))
        self.errors = tf.reduce_mean(tf.cast(mistakes, tf.float32))

def run_epoch(num_epochs=10):
    config = LMConfig()
    train_data, _, _, words, word_to_id = \
        ptb_raw_data('simple-examples/data')
    config.vocab_size = len(words)

    input_train = PTBInput(config, train_data)
    batch_len = input_train.batch_len
    model = PTBModel(config)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Start training...')
    for epoch in range(num_epochs):
        for i in range(batch_len):
            x_batch, y_batch = input_train.next_batch()

            feed_dict = {model._inputs: x_batch, model._targets: y_batch}
            sess.run(model.optim, feed_dict=feed_dict)

            if i % 500 == 0:
                cost = sess.run(model.cost, feed_dict=feed_dict)

                msg = "Epoch: {0:>3}, batch: {1:>5}, Loss: {2:>6.3}"
                print(msg.format(epoch + 1, i + 1, cost))

                pred = sess.run(model._pred, feed_dict=feed_dict)
                word_ids = sess.run(tf.argmax(pred, 1))
                print('Predicted:', ' '.join(words[w] for w in word_ids))
                true_ids = np.argmax(y_batch, 1)
                print('True:', ' '.join(words[w] for w in true_ids))

    print('Finish training...')
    sess.close()


if __name__ == '__main__':
    run_epoch()