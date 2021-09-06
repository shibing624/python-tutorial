# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 背包问题
"""


class Dp(object):
    def __init__(self, money):
        self.mark = [0 for _ in range(money + 1)]  # 备忘录
        self.money = money

    def dp(self, money):
        self.coin = 0  # 需要的硬币数为0
        if self.mark[money] != 0:  # 在备忘录中寻找该金额下的最少硬币找零数，若存在，则取出
            self.coin = self.mark[money]
        elif money <= 0:  # 边界问题
            if money == 0:  # 如果金额为零，则代表刚好算是一种找零方法
                self.coin = 0  # 这里的0不是代表硬币数为0，而是代表这种方法可行，因为在下面已经有加1，若是这里coin为1，结果就会比答案多1
            else:
                self.coin = float("inf")  # 若是金额为负数，即“拿多了”，这种方法不可行，则硬币消耗数为 无穷大
        elif money > 0:
            self.coin = min(self.dp(money - 1), self.dp(money - 3), self.dp(money - 5)) + 1  # 递归，找出最少的可以凑齐金额数money的方法
        self.mark[money] = self.coin  # 做备忘录
        return self.coin


def maxsum(inner_list):
    Max, temp = inner_list[0], 0
    for i in inner_list:
        if temp < 0:
            temp = i  # 更新起点
        else:
            temp += i
        Max = max(temp, Max)
    return Max


if __name__ == '__main__':
    m = Dp(65)  # 找零钱
    print(m.dp(10))
    print(maxsum([1, -2, 3, 10, -4, 7, 2, -5]))
