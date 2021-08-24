# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import threading
import time


def hello(name):
    print("hello %s\n" % name)

    global timer
    timer = threading.Timer(2.0, hello, ["Hawk"])
    timer.start()


def is_end(limit_time):
    timer = threading.Timer(limit_time, is_end, [limit_time])
    timer.start()
    return True


def end():
    return True


def timer_end():
    timer = threading.Timer(3, end)
    timer.start()


if __name__ == "__main__":
    # hello('girl')
    print(is_end(2))
