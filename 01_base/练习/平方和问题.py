'''两个数字的平方和是2022，请问这2个数分别是多少'''

for a in range(1,2022):
    if (2022 - a*a)**0.5 in range(1,2022):
        print(a)