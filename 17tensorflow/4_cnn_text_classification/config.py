# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/17
# Brief: 配置
import os


# data
dev_sample_percentage = 0.1  # percentage of the training data for validation
data_dir = "./data/zh_polarity"  # data file path

# model
embedding_dim = 128  # dimensionality of character embedding (default: 128)
filter_sizes = "3,4,5"  # comma-separated filter size (default: "3,4,5")
num_filters = 128  # number of filters per filter size
dropout_keep_prob = 0.5  # dropout keep probability
l2_reg_lambda = 0.0  # l2 regulaization lambda

# train
batch_size = 64  # batch size (default: 64)
num_epochs = 5  # number of training epochs (default: 5)
evaluate_every = 100  # evaluate model on dev set after this many steps (default: 100)
checkpoint_every = 100  # save model after this many steps (default: 100)
num_checkpoints = 5  # number of checkpoints to store

# proto
allow_soft_placement = True  # allow device soft device placement
log_device_placement = False  # log placement of ops on devices

infer_data_path = "./data/input_data.txt"  # infer data
checkpoint_dir = "./models/checkpoints"  # checkpoint directory from training run
eval_all_train_data = False  # evaluate on all training data

# directory to save the trained model
# create a new directory if the dir does not exist
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)