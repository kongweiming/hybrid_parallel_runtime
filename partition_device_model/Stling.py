#-*-encoding=utf-8-*-
import itertools
from itertools import product
import  numpy as np
#作者huangquan
def detail(x,y):
    l=[]
    def max_values(n,x):
        max_value=0
        iters=np.arange(0,x,1)
        for i in iters:
            max_value= np.power((n-x+2),i)*(n-x+1)+max_value
        return max_value
 
    def min_values(n,x):
        min_value=0
        iters=np.arange(0,x,1)
        for i in iters:
            min_value= np.power((n-x+2),i)*1+min_value
        return min_value
 
    def f(n,x):
        #n这里设置范围30以下，不够的朋友自己加
        a=range(1,30,1)
        b=[]
        while True:
            s=n//x
            y=n%x
            b=b+[y]
            if s==0:
                break
            n=s
        b.reverse()
        b=list(b)
        return  b
 
    a=list(set(x))
    iters=np.arange(1,a.__len__()+1,1)
    contianer=list()
    discontianer=list()
    count=0
    # for i in iters:
    for j in (np.arange(min_values(a.__len__(),y),max_values(a.__len__(),y)+1,1)):
        b=f(j,a.__len__()-y+2)
        if sum(b)==a.__len__() and (0 not in b):
            contianer.append(tuple(sorted(b)))
            discontianer=list(set(contianer))
    print("discontianer: ", discontianer)
 
    def strlingnumfun(a):
        finrestult=[]
        for i in np.arange(0,a.__len__(),1):
            z=[]
            listrestult=[]
            for j in np.arange(0,a[i].__len__(),1):
                result=list(itertools.combinations(list(np.arange(1,sum(a[i])+1,1)),a[i][j]))#a[1]修改为a[i]
                z.append(result)
            for x in product(*z):
                listrestult.append(list(x))
            for zz in listrestult:
                zzzs=[]
                for zzz in zz:
                    zzzz=list(zzz)
                    zzzs.extend(zzzz)
                    zzzs=list(set(zzzs))
 
                if zzzs.__len__()==sum(a[i]):#a[1]修改为a[i]
                    finrestult.append(zz)
        return finrestult
    result=strlingnumfun(discontianer)
    #print(result)
 
 
    for qq in result:
        qqs=sorted(qq)
        if qqs not in l:
            l.append(qqs)
 
 
 
    def S(n,m):
        if m>n or m==0:
            return  0
        if n==m:
            return 1
        if m==1:
            return 1
        return S(n-1,m-1)+S(n-1,m)*m
 
 
    def total(n,y):
        sumtotal=0
        # for i in np.arange(0,n+1,1):
        sumtotal=sumtotal+S(n,y)
        print("sumtotal: ",sumtotal)
 
    print("===========程序结果=============")
    for i in l:
        print(i)
    print ("==========结果个数==============")
    print(l.__len__())
    print("===========第二类斯特林验证程序是否合格============")
    total(sum(discontianer[0]),y)

    return l

# x=[1,2,5,6]
# print(detail(x))
 
#x的长度就是第二类斯特林数的输入，这里x=5（列表长度）
#x分为y组
# x=[1,2,5,6,10]
# y = 6
# x = [1,2,3,4,5,6,7,8]
# #print(detail(x,y))
# detail(x,y)