# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/18
# Brief: 训练model


import paddle.v2 as paddle

import config
import reader
from network import DSSM
from utils import load_dic, logger, ModelType, ModelArch, display_args


def train(train_data_paths=None,
          test_data_paths=None,
          source_dic_path=None,
          target_dic_path=None,
          model_arch=ModelArch.create_rnn(),
          batch_size=10,
          num_passes=10,
          share_semantic_generator=False,
          share_embed=False,
          class_num=2,
          num_workers=1,
          use_gpu=False):
    """
    train DSSM
    """
    default_train_paths = ["./data/classification/train/right.txt",
                           "./data/classification/train/wrong.txt"]
    default_test_paths = ["./data/classification/test/right.txt",
                          "./data/classification/test/wrong.txt"]
    default_dic_path = "./data/vocab.txt"
    layer_dims = [int(i) for i in config.config['dnn_dims'].split(',')]
    use_default_data = not train_data_paths
    if use_default_data:
        train_data_paths = default_train_paths
        test_data_paths = default_test_paths
        source_dic_path = default_dic_path
        target_dic_path = default_dic_path

    dataset = reader.Dataset(
        train_paths=train_data_paths,
        test_paths=test_data_paths,
        source_dic_path=source_dic_path,
        target_dic_path=target_dic_path
    )

    train_reader = paddle.batch(paddle.reader.shuffle(dataset.train, buf_size=1000),
                                batch_size=batch_size)
    test_reader = paddle.batch(paddle.reader.shuffle(dataset.test, buf_size=1000),
                               batch_size=batch_size)
    paddle.init(use_gpu=use_gpu, trainer_count=num_workers)

    # DSSM
    cost, prediction, label = DSSM(
        dnn_dims=layer_dims,
        vocab_sizes=[len(load_dic(path)) for path in [source_dic_path, target_dic_path]],
        model_arch=model_arch,
        share_semantic_generator=share_semantic_generator,
        class_num=class_num,
        share_embed=share_embed)()

    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    trainer = paddle.trainer.SGD(
        cost=cost,
        extra_layers=paddle.evaluator.auc(input=prediction, label=label),
        parameters=parameters,
        update_equation=adam_optimizer)

    feeding = {"source_input": 0, "target_input": 1, "label_input": 2}

    def _event_handler(event):
        """
        Define batch handler
        :param event:
        :return:
        """
        if isinstance(event, paddle.event.EndIteration):
            # output train log
            if event.batch_id % config.config['num_batches_to_log'] == 0:
                logger.info("Pass %d, Batch %d, Cost %f, %s" %
                            (event.pass_id, event.batch_id, event.cost, event.metrics))

            # test model
            if event.batch_id > 0 and event.batch_id % config.config['num_batches_to_test'] == 0:
                if test_reader is not None:
                    result = trainer.test(reader=test_reader, feeding=feeding)
                    logger.info("Test at Pass %d, %s" % (event.pass_id, result.metrics))

            # save model
            if event.batch_id > 0 and event.batch_id % config.config['num_batches_to_save_model'] == 0:
                model_desc = "classification_{arch}".format(arch=str(model_arch))
                with open("%sdssm_%s_pass_%05d.tar" %
                              (config.config['model_output_prefix'], model_desc,
                               event.pass_id), "w") as f:
                    parameters.to_tar(f)
                logger.info("save model: %sdssm_%s_pass_%05d.tar" %
                            (config.config['model_output_prefix'], model_desc, event.pass_id))

        # if isinstance(event, paddle.event.EndPass):
        #     result = trainer.test(reader=test_reader, feeding=feeding)
        #     logger.info("Test with pass %d, %s" % (event.pass_id, result.metrics))
        #     with open("./data/output/endpass/dssm_params_pass" + str(event.pass_id) + ".tar", "w") as f:
        #         parameters.to_tar(f)

    trainer.train(reader=train_reader,
                  event_handler=_event_handler,
                  feeding=feeding,
                  num_passes=num_passes)
    logger.info("training finish.")


if __name__ == '__main__':
    display_args(config.config)
    train(train_data_paths=config.config["train_data_paths"],
          test_data_paths=config.config["test_data_paths"],
          source_dic_path=config.config["source_dic_path"],
          target_dic_path=config.config["target_dic_path"],
          model_arch=ModelArch(config.config["model_arch"]),
          batch_size=config.config["batch_size"],
          num_passes=config.config["num_passes"],
          share_semantic_generator=config.config["share_network_between_source_target"],
          share_embed=config.config["share_embed"],
          class_num=config.config["class_num"],
          num_workers=config.config["num_workers"],
          use_gpu=config.config["use_gpu"])
