#-*- coding:utf-8 -*-

list1 = [i for i in range(0, 10)]
lenth = len( list1) 
for j in range(0,  lenth ):
    if list1[j] % 2 == 0: #如果list1[j]是一个偶数
        del list1[j] #那就删除list1[j]
print (list1)

