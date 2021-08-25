# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import multiprocessing
import time


def process(num):
    time.sleep(num)
    print('Process:', num)


if __name__ == '__main__':
    for i in range(5):
        p = multiprocessing.Process(target=process, args=(i,))
        p.start()

    print('CPU number:' + str(multiprocessing.cpu_count()))
    for p in multiprocessing.active_children():
        print('Child process name: ' + p.name + ' id: ' + str(p.pid))

    print('Process Ended')