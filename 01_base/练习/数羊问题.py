'''
编写一段代码，模拟我们数羊到入睡的过程：


每数五只羊，就提问一次：睡着了吗？
如果没有睡着，继续循环，并打印“继续数羊”。
如果睡着了，则停止循环，并打印“终于睡着了”。
'''


i = 0
while True:
    i += 1
    left_endpoint = 1 + 5 * ( i - 1 )
    right_endpoint = 1 + 5 * i
    for i in range(left_endpoint, right_endpoint):
        print(str(i)+'只羊')
    answer = input('睡着了吗？回答是或否：')
    if answer == '是':
        break
    print('继续数羊')
print('终于睡着了')

#方法二
睡觉的状态 = '还没睡'
a = 0
while 睡觉的状态 != '睡着': # 只要不是睡着，就继续数
    a +=1
    print(str(a)+'只羊')
    if a%9 == 0 : # %是取余数 每次数5只羊
        睡觉的状态 = input('睡着了嘛？')