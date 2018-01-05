# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/18
# Brief: read data set


def rnn_reader(file_path, word_dict, is_infer):
    """
    create reader for RNN, each line is a sample.

    :param file_path: file path.
    :param word_dict: vocab with content of '{word, id}',
                      'word' is string type , 'id' is int type.
    :return: data reader.
    """

    def reader():
        with open(file_path) as f:
            for line_id, line in enumerate(f):
                yield record_reader(line, word_dict, is_infer)

    return reader


def record_reader(line, word_dict, is_infer):
    """
    data format:
        <source words> [TAB] <target words> [TAB] <label>
    :param line:
    :param word_dict:
    :return:
    """
    fs = line.strip().split('\t')
    assert len(fs) == 3, "wrong format for rank\n" + \
                         "the format should be " + \
                         "<source words> [TAB] <target words> [TAB] <label>"

    left = sent2lm(fs[0], word_dict)
    right = sent2lm(fs[1], word_dict)
    if not is_infer:
        label = int(fs[2])
        return left[0], left[1], right[0], right[1], label
    return left[0], left[1], right[0], right[1]


def sent2lm(sent, word_dict):
    """
    transform a sentence to a list of language model ids.
    :param sent:
    :param word_dict:
    :return:
    """
    UNK = word_dict['<unk>']
    ids = [word_dict.get(w, UNK) for w in sent.strip().lower().split()] + [word_dict['<e>']]
    return ids[:-1], ids[1:]
