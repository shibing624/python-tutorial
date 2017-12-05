# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/18
# Brief: read data set
from utils import load_dict, logger, sent2lm


class Dataset(object):
    def __init__(self, train_path, test_path, word_dict, is_infer=False):
        self.train_path = train_path
        self.test_path = test_path
        self.word_dict = word_dict
        self.is_infer = is_infer

    def train(self):
        '''
        Load trainset.
        '''
        logger.info("[reader] load trainset from %s" % self.train_path)
        with open(self.train_path) as f:
            for line_id, line in enumerate(f):
                yield self.record_reader(line)

    def test(self):
        '''
        Load testset.
        '''
        with open(self.test_path) as f:
            for line_id, line in enumerate(f):
                yield self.record_reader(line)

    def infer(self):
        self.is_infer = True
        with open(self.train_path) as f:
            for line in f:
                yield self.record_reader(line)

    def record_reader(self, line):
        '''
        data format:
            <source words> [TAB] <target words> [TAB] <label>
        '''
        fs = line.strip().split('\t')
        assert len(fs) == 3, "wrong format for rank\n" + \
                             "the format should be " + \
                             "<source words> [TAB] <target words> [TAB] <label>"

        source = sent2lm(fs[0], self.word_dict)
        target = sent2lm(fs[1], self.word_dict)
        if not self.is_infer:
            label = int(fs[2])
            return source, target, label
        return source, target


if __name__ == "__main__":
    train_path = "./data/rank/train.txt"
    test_path = "./data/rank/test.txt"
    dic_path = "./data/vocab.txt"
    word_dict = load_dict(dic_path)
    dataset = Dataset(train_path, test_path, word_dict)
    for record in dataset.train():
        print(record)
