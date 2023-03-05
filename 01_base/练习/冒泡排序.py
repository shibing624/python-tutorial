'''
给定一个列表，请你对列表的元素进行     从大到小排序   与从小到大排序
'''


list1 = [13, 22, 6, 99, 11, 0]

for a in range(len(list1)):
    for b in range(a,len(list1)):
        if list1[a] < list1[b]:  #如果m大于了n
           list1[a] ,list1[b] =  list1[b],list1[a]#交换位置
print(list1)