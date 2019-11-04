# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import logging
import threading
import time


def get_logger():
    logger = logging.getLogger("threading_eg")
    logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler("out.log")
    fh = logging.StreamHandler()
    fmt = '%(asctime)s - %(name)s - %(processName)s - %(threadName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def doubler(number, logger):
    logger.debug("xxxx")
    logger.info("aaaa")
    logger.warning("bbbb")
    logger.error("cccc")

    result = number * 2
    time.sleep(5)
    logger.debug('yyyy: {}'.format(
        result))


logger = get_logger()
thread_names = ['Mike', 'George', 'Wanda', 'Dingbat', 'Nina']
for i in range(5):
    my_thread = threading.Thread(
        target=doubler, name=thread_names[i], args=(i, logger))
    my_thread.start()