import numpy as np
import pandas as pd
file = './irisx.csv'
df = pd.read_csv(file)
#print(df.head(10))
'''选取csv文件中的目标属性列，作为训练样本'''
y = df.iloc[0:100,5].values
y = np.where(y==' Setosa', -1, 1)
X = df.iloc[0:100,[1,2,3,4]].values
eta = 0.07
w = np.random.randn(16,4)
m = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],
             [0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],
             [1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],
             [1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]])

w1 = np.random.randn(4,4)
m1 = np.array([[0,0,0,1],[0,0,1,1],[0,1,1,1],[1,1,1,1]])
o1 = w1.dot(m1)
w_r = np.zeros(4)
itern = 100
def cal_wr():#计算最终的w
    w_r1 = np.zeros(4)
    for i in range(4):
        for j in range(4):
            w_r1[i] = w_r1[i]+o1[i][j]
    return w_r1

def output(X,w_r):
    r = X.dot(w_r.T)
    return r

def w_m():#w的状态
    o = w1.dot(m1)
    return o


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def acc(maxx,X):
    rig = 0
    for i in range(maxx):
        x = np.where(output(X[i],w_r)>=0,1,-1)
        if x == y[i]:
            rig = rig+1
    return rig/maxx


for l in range(itern):
    for i in range(100):
        #print(w1)
        o1 = w_m()
        w_r = cal_wr()
        out = output(X[i],w_r)
        #print("out",out)
        errors = y[i]-tanh(out)
        #print(errors)
        #print("----------")
        w1 = w1+eta*errors
o1 = w_m()
w_r = cal_wr()
acc = acc(100,X)

print(o1)
print(w_r)
print("准确率:",acc*100,"%")
