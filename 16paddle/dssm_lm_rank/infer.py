# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/18
# Brief: 预测


import os
import sys

import paddle.v2 as paddle

import config
import reader
from network import dssm_lm
from utils import logger, load_dict, load_reverse_dict


def infer(model_path, dic_path, infer_path, prediction_output_path, rnn_type="gru", batch_size=1):
    logger.info("begin to predict...")
    # check files
    assert os.path.exists(model_path), "trained model not exits."
    assert os.path.exists(dic_path), " word dictionary file not exist."
    assert os.path.exists(infer_path), "infer file not exist."

    logger.info("load word dictionary.")
    word_dict = load_dict(dic_path)
    word_reverse_dict = load_reverse_dict(dic_path)
    logger.info("dictionary size = %d" % (len(word_dict)))

    try:
        word_dict["<unk>"]
    except KeyError:
        logger.fatal("the word dictionary must contain <unk> token.")
        sys.exit(-1)

    # initialize PaddlePaddle
    paddle.init(use_gpu=config.use_gpu, trainer_count=config.num_workers)

    # load parameter
    logger.info("load model parameters from %s " % model_path)
    parameters = paddle.parameters.Parameters.from_tar(
        open(model_path, "r"))

    # load the trained model
    prediction = dssm_lm(
        vocab_sizes=[len(word_dict), len(word_dict)],
        emb_dim=config.emb_dim,
        hidden_size=config.hidden_size,
        stacked_rnn_num=config.stacked_rnn_num,
        rnn_type=rnn_type,
        share_semantic_generator=config.share_semantic_generator,
        share_embed=config.share_embed,
        is_infer=True)
    inferer = paddle.inference.Inference(
        output_layer=prediction, parameters=parameters)
    feeding = {"left_input": 0, "left_target": 1, "right_input": 2, "right_target": 3}

    logger.info("infer data...")
    # define reader
    reader_args = {
        "file_path": infer_path,
        "word_dict": word_dict,
        "is_infer": True,
    }
    infer_reader = paddle.batch(reader.rnn_reader(**reader_args), batch_size=batch_size)
    logger.warning("output prediction to %s" % prediction_output_path)
    with open(prediction_output_path, "w")as f:
        for id, item in enumerate(infer_reader()):
            left_text = " ".join([word_reverse_dict[id] for id in item[0][0]])
            right_text = " ".join([word_reverse_dict[id] for id in item[0][2]])
            probs = inferer.infer(input=item, field=["value"], feeding=feeding)
            f.write("%f\t%f\t%s\t%s" % (probs[0], probs[1], left_text, right_text))
            f.write("\n")


if __name__ == "__main__":
    infer(model_path=config.model_path,
          dic_path=config.dic_path,
          infer_path=config.infer_path,
          prediction_output_path=config.prediction_output_path,
          rnn_type=config.rnn_type)
