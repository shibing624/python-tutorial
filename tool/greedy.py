# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

# coding=utf-8
if __name__ == '__main__':
    beg = 50                       #背包50kg
    value = 0                      #已经获得的价值
    choice = []
    while beg > 0:                 #如果背包还有空位，则递归
        if beg >= 8:               #选择当前这一步的最优解，既选择B商品
            beg = beg - 8
            value = value + 13
            choice.append("B")
        elif beg >= 10:            #要是B商品选择不了，则选择第二单位价值的物品，即A物品
            beg = beg - 10
            value = value + 15
            choice.append("A")
        elif beg >= 6:
            beg = beg - 6
            value = value + 8
            choice.append("C")
        else:                      #当所有的物品都选择不了，则退出
            break
    print("剩余的背包重量：",beg)
    print("获得的总价值：",value)
    print("选择的物品的类型及顺序：",choice)