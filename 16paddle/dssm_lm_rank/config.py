# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/18
# Brief:
import os

################## for building word dictionary  ##################

max_word_num = 51200 - 2
cutoff_word_fre = 5

################## for training task  #########################
# path of training data
train_data_path = "data/rank/train.txt"
# path of testing data, if testing file does not exist,
# testing will not be performed at the end of each training pass
test_data_path = "data/rank/test.txt"
# path of word dictionary, if this file does not exist,
# word dictionary will be built from training data.
dic_path = "data/rank/vocab.txt"

share_semantic_generator = True  # whether to share network parameters between source and target
share_embed = True  # whether to share word embedding between source and target

num_workers = 1 # threads
use_gpu = False  # to use gpu or not

num_batches_to_log = 50
num_batches_to_save_model = 400  # number of batches to output model

# directory to save the trained model
# create a new directory if the directoy does not exist
model_save_dir = "output"

##################  for model configuration  ##################
rnn_type = "gru"  # "gru" or "lstm"
emb_dim = 256
hidden_size = 256
stacked_rnn_num = 2
batch_size = 32  # the number of training examples in one forward/backward pass
num_passes = 10  # how many passes to train the model

##################  for model infer  ##################
model_path = "output/dssm_pass_00009.tar"
infer_path = "data/rank/test.txt"
prediction_output_path  = "data/rank/pred.txt"

if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
