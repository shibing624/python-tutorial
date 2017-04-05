# -*- coding: utf-8 -*-
"""
@description: 生成器
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理


# 生成器使用 yield 关键字将值输出，
# 而迭代器则通过 next 的 return 将值返回；

# 与迭代器不同的是，生成器会自动记录当前的状态，
# 而迭代器则需要进行额外的操作来记录当前的状态。

# 之前的 collatz 猜想，简单循环的实现如下：
# collatz:
# 奇数 n：返回 3n + 1
# 偶数 n：返回 n / 2
# 直到 n 为 1 为止：

def collatz(n):
    sequence = []
    while n != 1:
        if n % 2 == 0:
            n /= 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence


for x in collatz(5):
    print(x)


# 生成器的版本如下：
def collatz(n):
    while n != 1:
        if n % 2 == 0:
            n /= 2
        else:
            n = 3 * n + 1
        yield n


for x in collatz(5):
    print(x)


# 迭代器的版本如下：
class Collatz(object):
    def __init__(self, start):
        self.value = start

    def __iter__(self):
        return self

    def next(self):
        if self.value == 1:
            raise StopIteration
        elif self.value % 2 == 0:
            self.value = self.value / 2
        else:
            self.value = 3 * self.value + 1
        return self.value


for x in collatz(5):
    print(x)

# 事实上，生成器也是一种迭代器：
x = collatz(5)
print(x)
# 它支持 next 方法，返回下一个 yield 的值：
print(x.next())
print(x.next())
# __iter__ 方法返回的是它本身：
print(x.__iter__())
