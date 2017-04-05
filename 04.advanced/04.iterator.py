# -*- coding: utf-8 -*-
"""
@description: 迭代器
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

x = [2, 4, 6]

for i in x:
    print(i)

# 列表好处是不需要对下标进行迭代，但是有些情况下，我们既希望获得下标，
# 也希望获得对应的值，那么可以将迭代器传给 enumerate 函数，
# 这样每次迭代都会返回一组 (index, value) 组成的元组：
x = [2, 4, 6]
for i, n in enumerate(x):
    print(i, 'is', n)


# 一个迭代器都有 __iter__ 、__iter__、next()这三个方法：

# 自定义一个 list 的取反迭代器：
class ReverseListIterator(object):
    def __init__(self, list):
        self.list = list
        self.index = len(list)

    def __iter__(self):
        return self

    def next(self):
        self.index -= 1
        if self.index >= 0:
            return self.list[self.index]
        else:
            raise StopIteration


x = range(10)
for i in ReverseListIterator(x):
    print(i)


# 只要我们定义了这三个方法，我们可以返回任意迭代值：

# 这里我们实现 Collatz 猜想：
#
# 奇数 n：返回 3n + 1
# 偶数 n：返回 n / 2
# 直到 n 为 1 为止：
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


for x in Collatz(5):
    print(x, )

# 不过迭代器对象存在状态，有问题：
i = Collatz(7)
for x, y in zip(i, i):
    print(x, y)


# 解决方法是将迭代器和可迭代对象分开处理，
# 这里提供了一个二分树的中序遍历实现：
class BinaryTree(object):
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __iter__(self):
        return InorderIterator(self)


class InorderIterator(object):
    def __init__(self, node):
        self.node = node
        self.stack = []

    def next(self):
        if len(self.stack) > 0 or self.node is not None:
            while self.node is not None:
                self.stack.append(self.node)
                self.node = self.node.left
            node = self.stack.pop()
            self.node = node.right
            return node.value
        else:
            raise StopIteration()


# 测试
tree = BinaryTree(
    left=BinaryTree(
        left=BinaryTree(1),
        value=2,
        right=BinaryTree(
            left=BinaryTree(3),
            value=4,
            right=BinaryTree(5)
        ),
    ),
    value=6,
    right=BinaryTree(
        value=7,
        right=BinaryTree(8)
    )
)

for value in tree:
    print(value)

# 不会出现之前的问题：
for x, y in zip(tree, tree):
    print(x, y)
