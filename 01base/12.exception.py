# -*- coding: utf-8 -*-
"""
@description: Try & Except
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

# 异常
# try & except 块
# 捕捉不同的错误类型
import math

while True:
    try:
        text = raw_input('>')
        if text[0] == 'q':
            break
        x = float(text)
        y = 1 / math.log10(x)
        print("1/log10({0}) = {1}".format(x, y))
    except ValueError:
        print("value must bigger than 0")
    except ZeroDivisionError:
        print("the value must not be 1")


# 自定义异常
# 异常是标准库中的类，这意味着我们可以自定义异常类：
class CommandError(ValueError):
    pass


valid_commands = {'start', 'stop', 'pause'}
while True:
    command = raw_input('>')
    if command == 'q':
        break
    try:
        if command.lower() not in valid_commands:
            raise CommandError('Invalid command: %s' % command)
    except CommandError:
        print("bad command string: %s" % command)

# finally
# try/catch 块还有一个可选的关键词 finally。

# 不管 try 块有没有异常， finally 块的内容总是会被执行，
# 而且会在抛出异常前执行，因此可以用来作为安全保证，
# 比如确保打开的文件被关闭。
try:
    print(1 / 0)
except ZeroDivisionError:
    print('divide by 0.')
finally:
    print('finally was called.')
