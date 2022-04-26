import numpy as np

# 用于记录变量的类，python里面好像没有静态变量这一说法，所以使用类来保存不变量
class StaticValue:
    def __init__(self):
        self.y1 = 0
        self.y2 = 0

    # 一些getter和setter函数
    def get_y1(self):
        return self.y1

    def set_y1(self,x):
        self.y1 = x

    def get_y2(self):
        return self.y2

    def set_y2(self,x):
        self.y2 = x

# 低通滤波器
def low_pass_filter(X,staticvalue):
    res = []
    for i in range(len(X)):
        if i < 6:
            res.append(lp(X[i],0,0,staticvalue))
        elif i <12:
            res.append(lp(X[i],X[i-6],0,staticvalue))
        else:
            res.append(lp(X[i],X[i-6],X[i-12],staticvalue))
    return res


# 高通滤波器
def high_pass_filter(X,staticvalue):
    res = []
    for i in range(len(X)):
        if i < 16:
            res.append(hp(X[i],0,0,staticvalue))
        elif i < 32:
            res.append(hp(X[i],X[i-16],0,staticvalue))
        else:
            res.append(hp(X[i],X[i-16],X[i-32],staticvalue))
    return res

# 低通滤波器
def lp(xt,xt_6,xt_12,staticvalue):
    y1 = staticvalue.get_y1()
    y2 = staticvalue.get_y2()
    res = 2*y1 - y2 + xt - 2*xt_6 + xt_12
    staticvalue.set_y2(y1)
    staticvalue.set_y1(res)
    return res
# 高通滤波器
def hp(xt,xt_16,xt_32,staticvalue):
    y1 = staticvalue.get_y1()
    res = 32*xt_16 - y1 - xt + xt_32
    staticvalue.set_y1(res)
    return res

# 5点平均
def derivative(X):
    res = []
    for i in range(len(X)):
        if i==0:
            res.append(1/8*(2*X[i+1]+X[i+2]))
        elif i==1:
            res.append(1/8*(-2*X[i-1]+2*X[i+1]+X[i+2]))
        elif i==len(X)-2:
            res.append(1/8*(-X[i-2]-2*X[i-1]+2*X[i+1]))
        elif i==len(X)-1:
            res.append(1/8*(-X[i-2]-2*X[i-1]))
        else:
            res.append(1/8*(-X[i-2]-2*X[i-1]+2*X[i+1]+X[i+2]))
    return res

# 滤波器的接口，将前面的函数集成起来
def filter(X):
    lp_static = StaticValue()
    hp_static = StaticValue()
    res = derivative(high_pass_filter(low_pass_filter(X,lp_static),hp_static))
    return res