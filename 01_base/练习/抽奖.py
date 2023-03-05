"""名单 = ['叶子','月娜','娜月','月叶','叶月','叶娜','月子','娜叶','娜子']
mychoice(名单)
打卡要求：风变科技计划抽一次年终奖
特等奖：1个组合课学习资格
一等奖：2个电饭煲
二等奖：4本python书
请你根据提示写出具有对应功能的代码"""


def mychoice(名单):
    import random
    中奖人员 = random.choice(名单)  # 随机抽取一个元素
    print(中奖人员)  # 打印出结果
    名单.remove(中奖人员)  # 从名单中移除中奖人员【这样一个人就只能抽到一次】


名单 = ['叶子', '月娜', '月娜2', '月娜3', '月娜4', '月娜5', '月娜6', '月娜7', '月娜8']

# for i in range(4): #奖品的等级
#     if i == 0: # 一等奖
#         print('恭喜以下同学获得：图书')
#         for i in range(2): #抽几个
#             mychoice(名单)
#     elif i == 1:   # 二等奖
#         print('恭喜以下同学获得：xxxx')
#         for i in range(5): #抽几个
#             mychoice(名单)
#     elif i == 2:# 三等奖
#         print('恭喜以下同学获得：XXX')
#         for i in range(3):#抽几个
#             mychoice(名单)
#     else:
#         print('恭喜以下同学获得：XXX')
#         for i in range(3):#抽几个
#             mychoice(名单)