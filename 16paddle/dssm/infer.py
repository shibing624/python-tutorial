# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/18
# Brief: 预测

import itertools
import paddle.v2 as paddle
import reader
from network import DSSM
from utils import logger, ModelArch, ModelType, load_dic
import config

paddle.init(use_gpu=False, trainer_count=1)


class Inferer(object):
    def __init__(self, model_path):
        logger.info("create DSSM model")
        self.source_dic_path = config.config["source_dic_path"]
        self.target_dic_path = config.config["target_dic_path"]
        dnn_dims = config.config["dnn_dims"]
        layer_dims = [int(i) for i in dnn_dims.split(',')]
        model_arch = ModelArch(config.config["model_arch"])
        share_semantic_generator = config.config["share_network_between_source_target"]
        share_embed = config.config["share_embed"]
        class_num = config.config["class_num"]
        prediction = DSSM(
            dnn_dims=layer_dims,
            vocab_sizes=[len(load_dic(path)) for path in [self.source_dic_path, self.target_dic_path]],
            model_arch=model_arch,
            share_semantic_generator=share_semantic_generator,
            class_num=class_num,
            share_embed=share_embed,
            is_infer=True)()

        # load parameter
        logger.info("load model parameters from %s " % model_path)
        self.parameters = paddle.parameters.Parameters.from_tar(
            open(model_path, "r"))
        self.inferer = paddle.inference.Inference(
            output_layer=prediction, parameters=self.parameters)

    def infer(self, data_path):
        logger.info("infer data...")
        dataset = reader.Dataset(train_paths=data_path,
                                 test_paths=None,
                                 source_dic_path=self.source_dic_path,
                                 target_dic_path=self.target_dic_path)
        infer_reader = paddle.batch(dataset.infer, batch_size=1000)
        prediction_output_path = config.config["prediction_output_path"]
        logger.warning("write prediction to %s" % prediction_output_path)
        with open(prediction_output_path, "w")as f:
            for id, batch in enumerate(infer_reader()):
                res = self.inferer.infer(input=batch)
                prediction = [" ".join(map(str, x)) for x in res]
                assert len(batch) == len(prediction), ("predict error, %d inputs,"
                                                       "but %d predictions") % (len(batch), len(prediction))
                f.write("\n".join(map(str, prediction)) + "\n")


if __name__ == '__main__':
    model_path = config.config["model_path"]
    infer_data_paths = config.config["infer_data_paths"]
    inferer = Inferer(model_path)
    inferer.infer(infer_data_paths)