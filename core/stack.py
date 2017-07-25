# -*- coding: utf-8 -*-

"""
@author: XuMing <shibing624@126.com>
@summary:
"""

stack = []
def pushit():
    stack.append(input(' enter new string: ').strip())

def popit():
    if len(stack) == 0:
        print("can not pop from an empty stack.")
    else:
        print("remove [", stack.pop(), ']')

def viewstack():
    print(stack)

if __name__ == '__main__':
    pushit()
    pushit()
    viewstack()
    popit()
    viewstack()