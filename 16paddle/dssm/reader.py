# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/18
# Brief: read data set
from utils import UNK, ModelType, TaskType, load_dic, \
    sent2ids, logger


class Dataset(object):
    def __init__(self, train_paths, test_paths, source_dic_path, target_dic_path):
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.source_dic_path = source_dic_path
        self.target_dic_path = target_dic_path

        self.source_dic = load_dic(self.source_dic_path)
        self.target_dic = load_dic(self.target_dic_path)

        self.record_reader = self._read_classification_record
        self.is_infer = False

    def train(self):
        """
        Load train data set
        :return:
        """
        logger.info("[reader] load train data set from %s" % ";".join(self.train_paths))
        for train_path in self.train_paths:
            label = 0
            if "right" in train_path:
                label = 1
            with open(train_path) as f:
                for line_id, line in enumerate(f):
                    yield self.record_reader(line, label=label)

    def test(self):
        """
        Load test data set
        :return:
        """
        for test_path in self.test_paths:
            label = 0
            if "right" in test_path:
                label = 1
            with open(test_path) as f:
                for line_id, line in enumerate(f):
                    yield self.record_reader(line, label=label)

    def infer(self):
        self.is_infer = True
        for train_path in self.train_paths:
            with open(train_path) as f:
                for line in f:
                    yield self.record_reader(line)

    def _read_classification_record(self, line, label=0):
        """
        data format:
            label.txt
            <source words> [tab] <target words>
        :param line:
        :return:
        """
        parts = line.strip().split("\t")
        assert len(parts) == 2, "wrong format for classification\n" + \
                                "the format is: <source words> [tab] <target words>"
        source = sent2ids(parts[0], self.source_dic)
        target = sent2ids(parts[1], self.target_dic)
        if not self.is_infer:  # train or test
            return source, target, label
        return source, target


if __name__ == "__main__":
    train_paths = ["./data/classification/train/right.txt",
                   "./data/classification/train/wrong.txt"]
    test_paths = ["./data/classification/test/right.txt",
                  "./data/classification/test/wrong.txt"]
    source_dic = "./data/vocab.txt"
    dataset = Dataset(train_paths, test_paths, source_dic, source_dic)
    for record in dataset.train():
        print(record)
