# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/16
# Brief: 
config = {
    "qa_examples_file": "../data/qa.examples.train.e2e.top10.filter.tsv",
    "word_embeddings_file": "../data/word_embedding/glove.6B.100d.txt",
    "vocabulary_size": 400000,
    "embedding_size": 100,
    "num_classes": 6,
    "filter_sizes": [3, 4],
    "num_filters": 4,
    "dropout_keep_prob": 0.85,
    "embeddings_trainable": True,
    "total_iter": 100000,
    "batch_size": 400,
    "val_size": 400,
    "l2_reg_lambda": 0.1
}
