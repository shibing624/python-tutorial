# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/11/13
# Brief:
import time

def gcd(pair):
    a, b = pair
    low = min(a, b)
    for i in range(low, 0, -1):
        if a % i == 0 and b % i == 0:
            return i


numbers = [(1963309, 2265973), (2012332, 2300345), (1551645, 22341234), (89765454, 12345678),
           (1963300, 1265900), (23012355, 2300345), (4551645, 22341234), (29765458, 12345678)]


def cal():
    return list(map(gcd, numbers))

def cal2():
    from concurrent.futures.thread import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=4)
    result = list(pool.map(gcd,numbers))
    return result

def cal3():
    from concurrent.futures import ProcessPoolExecutor
    import os
    print(os.cpu_count())
    max_workers = (os.cpu_count() or 1) * 5
    print(max_workers)
    pool = ProcessPoolExecutor(max_workers=4)
    result = list(pool.map(gcd,numbers))
    return result

start = time.time()
ret=cal()
print(ret)
end = time.time()
print("spend time: %.3f s" % (end - start))

start = time.time()
ret = cal2()
print(ret)
end = time.time()
print("spend time: %.3f s" % (end - start))

start = time.time()
ret=cal3()
print(ret)
end = time.time()
print("spend time: %.3f s" % (end - start))