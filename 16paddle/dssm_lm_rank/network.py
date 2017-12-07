# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Brief: 定义dssm网络结构

import paddle.v2 as paddle
from paddle.v2.attr import ParamAttr

from utils import logger


def dssm_lm(vocab_sizes=[],
            emb_dim=256,
            hidden_size=256,
            stacked_rnn_num=2,
            rnn_type="gru",
            share_semantic_generator=False,
            share_embed=False,
            is_infer=False):
    """
    init dssm network
    :param vocab_sizes: 2d tuple (size of both left and right items.)
    :param share_semantic_generator: bool (whether to share the semantic vector generator for both left and right.)
    :param share_embed: bool (whether to share the embeddings between left and right.)
    :param is_infer: inference
    """
    assert len(vocab_sizes) == 2, "vocab sizes specify the sizes left and right inputs, dim is 2."
    logger.info("vocabulary sizes: %s" % str(vocab_sizes))
    # input layers
    left_input = paddle.layer.data(name="left_input", type=paddle.data_type.integer_value_sequence(vocab_sizes[0]))
    right_input = paddle.layer.data(name="right_input", type=paddle.data_type.integer_value_sequence(vocab_sizes[1]))

    # target
    left_target = paddle.layer.data(name="left_target", type=paddle.data_type.integer_value_sequence(vocab_sizes[0]))
    right_target = paddle.layer.data(name="right_target", type=paddle.data_type.integer_value_sequence(vocab_sizes[1]))

    # rank label
    if not is_infer:
        label = paddle.layer.data(name="label", type=paddle.data_type.integer_value(1))

    # share params
    prefixs = '_ _'.split() if share_semantic_generator else 'left right'.split()
    embed_prefixs = '_ _'.split() if share_embed else 'left right'.split()

    # embedding layer
    word_vecs = []
    for id, input in enumerate([left_input, right_input]):
        x = create_embedding(input, emb_dim=emb_dim, prefix=embed_prefixs[id])
        word_vecs.append(x)

    # rnn layer
    features = []
    for id, input in enumerate(word_vecs):
        if rnn_type == "lstm":
            x = create_lstm(input, hidden_size=hidden_size, stacked_rnn_num=stacked_rnn_num, prefix=prefixs[id])
        elif rnn_type == "gru":
            x = create_gru(input, hidden_size=hidden_size, stacked_rnn_num=stacked_rnn_num, prefix=prefixs[id])
        features.append(x)

    # fc and output layer
    assert len(features) == 2, "dim must be 2"
    left_output = paddle.layer.fc(input=[features[0]], size=vocab_sizes[0], act=paddle.activation.Softmax())
    right_output = paddle.layer.fc(input=[features[1]], size=vocab_sizes[1], act=paddle.activation.Softmax())

    # perplexity
    left_entropy = paddle.layer.cross_entropy_cost(input=left_output, label=left_target)
    right_entropy = paddle.layer.cross_entropy_cost(input=right_output, label=right_target)

    # pooling to sum/avg score
    left_score = paddle.layer.pooling(input=left_entropy, pooling_type=paddle.pooling.Sum())
    right_score = paddle.layer.pooling(input=right_entropy, pooling_type=paddle.pooling.Sum())

    # cost
    if not is_infer:
        cost = paddle.layer.rank_cost(left_score, right_score, label=label)
        return cost, label
    # infer
    return left_score, right_score


def create_embedding(input, emb_dim=256, prefix=""):
    """
    A word embedding vector layer
    :param input:
    :param emb_dim:
    :param prefix:
    :return:
    """
    logger.info("create embedding table [%s] which dim is %d" % (prefix, emb_dim))
    emb = paddle.layer.embedding(input=input, size=emb_dim, param_attr=ParamAttr(name='%s_emb.w' % prefix))
    return emb


def create_gru(emb, hidden_size=256, stacked_rnn_num=2, prefix=''):
    """
    A GRU sentence vector learner.
    :param emb:
    :param hidden_size:
    :param stacked_rnn_num:
    :param prefix:
    :return:
    """
    logger.info("create gru")
    for i in range(stacked_rnn_num):
        rnn_cell = paddle.networks.simple_gru(input=rnn_cell if i else emb, size=hidden_size)
    return rnn_cell


def create_lstm(emb, hidden_size=256, stacked_rnn_num=2, prefix=''):
    """
    A LSTM sentence vector learner.
    :param emb:
    :param hidden_size:
    :param stacked_rnn_num:
    :param prefix:
    :return:
    """
    logger.info("create lstm")
    for i in range(stacked_rnn_num):
        rnn_cell = paddle.networks.simple_lstm(input=rnn_cell if i else emb, size=hidden_size)
    return rnn_cell
