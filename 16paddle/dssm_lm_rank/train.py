# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/18
# Brief: 训练model


import os
import paddle.v2 as paddle

import config
import reader
from network import dssm_lm
from utils import logger, build_dict, load_dict, display_args


def train(train_data_path=None,
          test_data_path=None,
          word_dict=None,
          batch_size=10,
          num_passes=10,
          share_semantic_generator=False,
          share_embed=False,
          num_workers=1,
          use_gpu=False):
    """
    train DSSM
    """
    dataset = reader.Dataset(train_data_path, test_data_path, word_dict)

    train_reader = paddle.batch(paddle.reader.shuffle(dataset.train, buf_size=102400),
                                batch_size=batch_size)
    test_reader = paddle.batch(paddle.reader.shuffle(dataset.test, buf_size=65536),
                               batch_size=batch_size)
    # initialize PaddlePaddle
    paddle.init(use_gpu=use_gpu, trainer_count=num_workers)

    # DSSM
    cost, label = dssm_lm(
        vocab_sizes=[len(word_dict), len(word_dict)],
        emb_dim=config.emb_dim,
        hidden_size=config.hidden_size,
        stacked_rnn_num=config.stacked_rnn_num,
        rnn_type=config.rnn_type,
        share_semantic_generator=share_semantic_generator,
        share_embed=share_embed)()

    # create parameters
    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5, max_average_window=10000))
    # create trainer
    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=adam_optimizer)

    feeding = {"source_input": 0, "target_input": 1, "label_input": 2}

    # define the event_handler callback
    def _event_handler(event):
        """
        Define batch handler
        :param event:
        :return:
        """
        if isinstance(event, paddle.event.EndIteration):
            # output train log
            if event.batch_id % config.num_batches_to_log == 0:
                logger.info("Pass %d, Batch %d, Cost %f, %s" %
                            (event.pass_id, event.batch_id, event.cost, event.metrics))

            # save model
            if event.batch_id > 0 and event.batch_id % config.num_batches_to_save_model == 0:
                model_desc = "rank_{arch}".format(arch=str(config.rnn_type))
                save_name = os.path.join(config.model_save_dir, "dssm_%s_pass_%05d.tar" %
                                         (model_desc, event.pass_id))
                with open(save_name, "w") as f:
                    parameters.to_tar(f)
                logger.info("save model: dssm_%s_pass_%05d.tar" %
                            (model_desc, event.pass_id))

        if isinstance(event, paddle.event.EndPass):
            if test_reader is not None:
                result = trainer.test(reader=test_reader)
                logger.info("Test with Pass %d, %s" %
                            (event.pass_id, result.metrics))
            save_name = os.path.join(config.model_save_dir, "dssm_pass_%05d.tar" % event.pass_id)
            with open(save_name, "w") as f:
                parameters.to_tar(f)

    trainer.train(reader=train_reader,
                  event_handler=_event_handler,
                  # feeding=feeding,
                  num_passes=num_passes)
    logger.info("training finish.")


def main():
    # prepare vocab
    if not (os.path.exists(config.dic_path) and
                os.path.getsize(config.dic_path)):
        logger.info(("word dictionary does not exist, "
                     "build it from the training data"))
        build_dict(config.train_data_path, config.dic_path, config.max_word_num,
                   config.cutoff_word_fre)
    logger.info("load word dictionary.")
    word_dict = load_dict(config.dic_path)
    logger.info("dictionary size = %d" % (len(word_dict)))

    train(train_data_path=config.train_data_path,
          test_data_path=config.test_data_path,
          word_dict=word_dict,
          batch_size=config.batch_size,
          num_passes=config.num_passes,
          share_semantic_generator=config.share_network_between_source_target,
          share_embed=config.share_embed,
          num_workers=config.num_workers,
          use_gpu=config.use_gpu)


if __name__ == "__main__":
    main()
