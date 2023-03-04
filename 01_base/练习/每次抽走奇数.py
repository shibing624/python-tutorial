'''
要求：0-100，每次抽走奇数，打印剩余的那个数字
'''

aList = []
for i in range(0,2023):
    aList.append(i)

while len(aList)>1:
    aList = aList[1::2]
    print(aList)
print(aList)