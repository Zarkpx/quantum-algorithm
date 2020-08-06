from pyqpanda import *
import pyqpanda.pyQPanda as pq
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def qfg(qlist):
    circ = QCircuit()
    
    qnum = len(qlist)
    for i in range(0,qnum):
        circ.insert(H(qlist[qnum-1-i]))
        for j in range(i+1,qnum):
            circ.insert(CR(qlist[qnum-1-j],qlist[qnum-1-i],math.pi/(1<<(j-i))))
            
    for i in range(0,qnum//2):
        circ.insert(CNOT(qlist[i],qlist[qnum-1-i]))
        circ.insert(CNOT(qlist[qnum-1-i],qlist[i])) 
        circ.insert(CNOT(qlist[i],qlist[qnum-1-i]))
        
    return circ

def plotBar(xdata,ydata):        
    plt.rcParams['font.sans-serif']=['Arial']
    plt.title("Origin Q",loc="right",alpha=0.5)
    plt.ylabel("prob")
    plt.xlabel("States")
    
    rects1 = plt.bar(x=xdata,height=ydata,width=0.8,color='blue')
    plt.ylim(0,1)
    #plt.xticks(xdata)
    plt.xticks(rotation=60)
    
    #for rect in rects1:
        #height = rect.get_height()
        #plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center",va="bottom")
    
    plt.show()

init(QMachineType.CPU)
qubits = qAlloc_many(4)
cbits = cAlloc_many(4)



#print(len(qubits))

# 构建量子程序
prog = QProg()
prog.insert(qfg(qubits)).insert(measure_all(qubits,cbits))

result1 = run_with_configuration(prog,cbits,100)
result2 = directly_run(prog)
# 获得目标量子比特的概率测量结果，其对应的下标为二进制。
result3 = prob_run_dict(prog,qubits,-1)
#获得目标量子比特的概率测量结果， 其对应的下标为十进制。
result4 = prob_run_tuple_list(prog,qubits,-1)
#获得目标量子比特的概率测量结果， 其对应的下标为二进制。
result5 = prob_run_list(prog,qubits,-1)

print("量子态出现次数：",result1)
print("-----------------------")
print("测量结果存储至经典寄存器：",result2)
print("-----------------------")
print("概率测量二进制：",result3)

xdata=list(result3.keys())
ydata=list(result3.values())
print(xdata)
print(ydata)
plotBar(xdata,ydata)

pq.draw_qprog(prog)

