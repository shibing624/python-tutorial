# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/18
# Brief: 工具类

import logging
import paddle

UNK = 0

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def model_attr_name(model):
    return model.upper() + "_MODE"


def create_attrs(cls):
    for id, mode in enumerate(cls.modes):
        setattr(cls, model_attr_name(mode), id)


def make_check_method(cls):
    """
    create methods for classes
    :param cls:
    :return:
    """

    def method(mode):
        def _method(self):
            return self.mode == getattr(cls, model_attr_name(mode))

        return _method

    for id, mode in enumerate(cls.modes):
        setattr(cls, "is_" + mode, method(mode))


def make_create_method(cls):
    def method(mode):
        @staticmethod
        def _method():
            key = getattr(cls, model_attr_name(mode))
            return cls(key)

        return _method

    for id, mode in enumerate(cls.modes):
        setattr(cls, "create_" + mode, method(mode))


def make_str_method(cls, type_name="unk"):
    def _str_(self):
        for mode in cls.modes:
            if self.mode == getattr(cls, model_attr_name(mode)):
                return mode

    def _hash_(self):
        return self.mode

    setattr(cls, '__str__', _str_)
    setattr(cls, '__repr__', _str_)
    setattr(cls, '__hash__', _hash_)
    cls.__name__ = type_name


def _init_(self, mode, cls):
    if isinstance(mode, int):
        self.mode = mode
    elif isinstance(mode, cls):
        self.mode = mode.mode
    else:
        raise Exception("wrong mode type, get type: %s, value: %s" %
                        (type(mode), mode))


def build_mode_class(cls):
    create_attrs(cls)
    make_str_method(cls)
    make_check_method(cls)
    make_create_method(cls)


class TaskType:
    modes = 'train test infer'.split()

    def __init__(self, mode):
        _init_(self, mode, TaskType)


class ModelType:
    modes = 'classification rank regression'.split()

    def __init__(self, mode):
        _init_(self, mode, ModelType)


class ModelArch:
    modes = 'rnn fc cnn'.split()

    def __init__(self, mode):
        _init_(self, mode, ModelArch)


build_mode_class(TaskType)
build_mode_class(ModelType)
build_mode_class(ModelArch)


def sent2ids(sent, vocab):
    """
    transform a sentence to a list of ids
    :param sent: a sentence
    :param vocab: a word dict
    :return: list
    """
    return [vocab.get(w, UNK) for w in sent.split()]


def load_dic(path):
    """
    word dic format:one line a word
    :param path:
    :return: dict
    """
    dic = {}
    if not path:
        logger.error("path is wrong. please check word dic path.")
    with open(path) as f:
        for id, line in enumerate(f):
            word = line.strip()
            dic[word] = id
    return dic


def display_args(args):
    logger.info("config:")
    for k, v in args.items():
        logger.info("{}: {}".format(k, v))


if __name__ == '__main__':
    t = TaskType(2)
    print(t) # infer
    t = TaskType.create_train()
    print(t)
    print("is", t.is_train())
    k = ModelArch.create_rnn()
    print(k)
