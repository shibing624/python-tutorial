# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/16
# Brief: cnn网络结构

import csv
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import config
import data_helpers
from util import to_categorical

if config.eval_all_train_data:
    x_raw, y = data_helpers.load_data_labels(config.data_dir)
    y = to_categorical(y)
    y = np.argmax(y, axis=1)
elif config.infer_data_path:
    infer_datas = list(open(config.infer_data_path, "r", encoding="utf-8").readlines())
    infer_datas = [s.strip() for s in infer_datas]
    x_raw = data_helpers.load_infer_data(infer_datas)
    y = []
else:
    x_raw = data_helpers.load_infer_data(
        ["do you think it is right.", "everything is off.", "i hate you .", "it is a bad film.",
         "good man and bad person.", "价格不是最便宜的，招商还是浦发银行是238*12=2856.00人家还可以分期的。",
         u"驱动还有系统要自装,还有显卡太鸡巴低了.还有装系统太麻烦了"
         ])
    y = [1, 0, 0, 0, 1, 0, 1]

# map data into vocabulary
checkpoint_dir = config.checkpoint_dir
vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
print("vocab_path:", vocab_path)
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvluating...\n")

checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
print("checkpoint file", checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=config.allow_soft_placement,
                                  log_device_placement=config.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # get the placeholders
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), config.batch_size, 1, shuffle=False)

        # collect the predictions
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# print accuracy if y_test is defined
if y is not None and len(y) > 0:
    correct_predictions = float(sum(all_predictions == y))
    print("Total number of test examples: {}".format(len(y)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y))))

# save the evaluation to csv
x_raw = [x.encode("utf-8") for x in x_raw]
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
print(predictions_human_readable)
with open(out_path, "w")as f:
    csv.writer(f).writerows(predictions_human_readable)
