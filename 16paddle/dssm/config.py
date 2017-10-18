# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/18
# Brief: 

config = {
    "train_data_paths": ["./data/classification/train/right.txt",
                         "./data/classification/train/wrong.txt"],  # path of training dataset
    "test_data_paths": ["./data/classification/test/right.txt",
                        "./data/classification/test/wrong.txt"],  # # path of testing dataset
    "source_dic_path": "./data/vocab.txt",  # path of the source's word dictionary
    "target_dic_path": "./data/vocab.txt",  # path of the target's word dictionary
    "model_arch": 0,  # 0 is rnn; 1 is fc; 2 is cnn
    "batch_size": 10,  # size of mini-batch (default:10)
    "num_passes": 10,  # number of passes to run(default:10)
    "share_network_between_source_target": False,  # whether to share network parameters between source and target
    "share_embed": False,  # share word embedding between source and target
    "dnn_dims": "256,128,64,32",  # dimentions of dnn layers, default is '256,128,64,32'
    "num_workers": 4,  # num worker threads, default 1
    "use_gpu": False,  # use GPU devices
    "class_num": 2,  # number of categories for classification task
    "model_output_prefix": "./output/",  # prefix of the path for model to store
    "num_batches_to_log": 100,  # log
    "num_batches_to_test": 200,  # batches to test
    "num_batches_to_save_model": 400,  # number of batches to output model, (default: 400)
}
