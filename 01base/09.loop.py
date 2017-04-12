# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

# while 循环
# while <condition>:
#     <statesments>
# Python会循环执行<statesments>，直到<condition>不满足为止。
i = 0
total = 0
while i <= 100:
    total += i
    i += 1
print(total)  # 5050

# 空容器会被当成False，因此可以用while循环读取容器的所有元素
plays = set(['Hamlet', 'Mac', 'King'])
while plays:
    play = plays.pop()
    print('Perform', play)

# for 循环
total = 0
for i in range(100000):
    total += i
print(total)  # 4999950000

# 然而这种写法有一个缺点：在循环前，它会生成一个长度为 100000 的临时列表。
# 生成列表的问题在于，会有一定的时间和内存消耗，当数字从 100000 变得更大时，
# 时间和内存的消耗会更加明显。

# 为了解决这个问题，我们可以使用 xrange 来代替 range 函数，
# 其效果与range函数相同，但是 xrange 并不会一次性的产生所有的数据：
total = 0
for i in xrange(100000):
    total += i
print(total)  # 4999950000

# continue 语句
# 遇到 continue 的时候，程序会返回到循环的最开始重新执行。
values = [7, 6, 4, 7, 19, 2, 1]
for i in values:
    if i % 2 != 0:
        # 忽略奇数
        continue
    print(i)
# 6
# 4
# 2

# break 语句
# 遇到 break 的时候，程序会跳出循环，不管循环条件是不是满足
command_list = ['start',
                '1',
                '2',
                '3',
                '4',
                'stop',
                'restart',
                '5',
                '6']
while command_list:
    command = command_list.pop(0)
    if command == 'stop':
        break
    print(command)
# start
# 1
# 2
# 3
# 4
