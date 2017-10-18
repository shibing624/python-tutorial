# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/18
# Brief: 定义dssm网络结构

from paddle import v2 as paddle
from paddle.v2.attr import ParamAttr
from utils import TaskType, logger, ModelArch, ModelType


class DSSM(object):
    def __init__(self, dnn_dims=[],
                 vocab_sizes=[],
                 model_type=ModelType.create_classification(),
                 model_arch=ModelArch.create_rnn(),
                 share_semantic_generator=False,
                 class_num=2,
                 share_embed=False,
                 is_infer=False
                 ):
        """
        init dssm network
        :param dnn_dims: list of int (dimentions of each layer in semantic vector generator.)
        :param vocab_sizes: 2d tuple (size of both left and right items.)
        :param model_type: classification
        :param model_arch: model architecture
        :param share_semantic_generator: bool (whether to share the semantic vector generator for both left and right.)
        :param class_num: number of categories.
        :param share_embed: bool (whether to share the embeddings between left and right.)
        :param is_infer: inference
        """
        assert len(vocab_sizes) == 2, ("vocab sizes specify the sizes left and right inputs, dim is 2.")
        assert len(dnn_dims) > 1, "more than two layers is needed."

        self.dnn_dims = dnn_dims
        self.vocab_sizes = vocab_sizes
        self.share_semantic_generator = share_semantic_generator
        self.share_embed = share_embed
        self.model_type = ModelType(model_type)
        self.model_arch = ModelArch(model_arch)
        self.class_num = class_num
        self.is_infer = is_infer
        logger.warning("build DSSM model with config of %s, %s" %
                       (self.model_type, self.model_arch))
        logger.info("vocabulary sizes: %s" % str(self.vocab_sizes))

        _model_arch = {
            "rnn": self.create_rnn,
            "cnn": self.create_cnn,
            "fc": self.create_fc,
        }

        def _model_arch_creater(emb, prefix=""):
            sent_vec = _model_arch.get(str(model_arch))(emb, prefix)
            dnn = self.create_dnn(sent_vec, prefix)
            return dnn

        self.model_arch_creater = _model_arch_creater
        self.model_type_creater = self._build_classification_model

    def __call__(self):
        return self._build_classification_model()

    def create_embedding(self, input, prefix=""):
        logger.info("create embedding table [%s] which dim is %d" % (prefix, self.dnn_dims[0]))
        emb = paddle.layer.embedding(input=input, size=self.dnn_dims[0], param_attr=ParamAttr(name='%s_emb.w' % prefix))
        return emb

    def create_rnn(self, emb, prefix=""):
        """
        A gru sentence vector learner
        :param emb:
        :param prefix:
        :return:
        """
        gru = paddle.networks.simple_gru(input=emb, size=256)
        sent_vec = paddle.layer.last_seq(gru)
        return sent_vec

    def create_cnn(self, emb, prefix=""):
        """
        A muli-layer CNN
        :param emb: output of the embedding layer
        :param prefix: prefix of layers' names
        :return:
        """

        def create_conv(context_len, hidden_size, prefix):
            key = "%s_%d_%d" % (prefix, context_len, hidden_size)
            conv = paddle.networks.sequence_conv_pool(
                input=emb,
                context_len=context_len,
                hidden_size=hidden_size,
                context_proj_param_attr=ParamAttr(name=key + 'contex_proj.w'),
                fc_param_attr=ParamAttr(name=key + '_fc.w'),
                fc_bias_attr=ParamAttr(name=key + '_fc.b'),
                pool_bias_attr=ParamAttr(name=key + '_pool.b')
            )
            return conv

        logger.info("create seq conv pool context width is 3")
        conv_3 = create_conv(3, self.dnn_dims[1], "cnn")
        logger.info("create seq conv pool context width is 4")
        conv_4 = create_conv(4, self.dnn_dims[1], "cnn")
        return conv_3, conv_4

    def create_fc(self, emb, prefix=''):
        """
        A multi-layer fully connected neural networks
        :param emb: output of the embedding layer
        :param prefix: prefix of layers' names
        :return:
        """
        _input_layer = paddle.layer.pooling(
            input=emb, pooling_type=paddle.pooling.Max())
        fc = paddle.layer.fc(input=_input_layer, size=self.dnn_dims[1])
        return fc

    def create_dnn(self, sent_vec, prefix):
        # if more than 3 layers, add a fc layer
        if len(self.dnn_dims) > 1:
            _input_layer = sent_vec
            for id, dim in enumerate(self.dnn_dims[1:]):
                name = "%s_fc_%d_%d" % (prefix, id, dim)
                logger.info("create fc layer [%s] which dim is %d" % (name, dim))
                fc = paddle.layer.fc(
                    name=name,
                    input=_input_layer,
                    size=dim,
                    act=paddle.activation.Tanh(),
                    param_attr=ParamAttr(name='%s.w' % name),
                    bias_attr=ParamAttr(name='%s.b' % name)
                )
                _input_layer = fc
        return _input_layer

    def _build_classification_model(self):
        logger.info("build classification model")
        assert self.class_num > 0, "num of class need more than zero."
        source = paddle.layer.data(name="source_input",
                                   type=paddle.data_type.integer_value_sequence(self.vocab_sizes[0]))
        target = paddle.layer.data(name="target_input",
                                   type=paddle.data_type.integer_value_sequence(self.vocab_sizes[1]))
        label = paddle.layer.data(name="label_input",
                                  type=paddle.data_type.integer_value(self.class_num))
        prefixs = '_ _'.split() if self.share_semantic_generator else 'left right'.split()
        embed_prefixs = '_ _'.split() if self.share_embed else 'left right'.split()

        word_vecs = []
        for id, input in enumerate([source, target]):
            embed = self.create_embedding(input, prefix=embed_prefixs[id])
            word_vecs.append(embed)

        semantics = []
        for id, input in enumerate(word_vecs):
            nn = self.model_arch_creater(input, prefix=prefixs[id])
            semantics.append(nn)

        # classification
        concated_vector = paddle.layer.concat(semantics)
        prediction = paddle.layer.fc(input=concated_vector, size=self.class_num, act=paddle.activation.Softmax())
        cost = paddle.layer.classification_cost(input=prediction, label=label)

        if not self.is_infer:
            return cost, prediction, label
        return prediction
