import math

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import random as rd

pi = math.pi


#评价标准，类内及类间距离
def distance(center,X,cls_num):
    cd_total = 0
    c_d = []
    item_cls = []
    cd_min = 0
    for m in X:
        for i in range(cls_num):
            c_d.append((m[0]-center[i][0])**2+(m[1]-center[i][1])**2+(m[2]-center[i][2])**2+(m[3]-center[i][3])**2)
        cd_min = min(c_d)
        cls = c_d.index(cd_min)
        item_cls.append(cls)
        cd_total = cd_total+cd_min
        c_d = []
    #类间距离
    c1 = (center[0][0]-center[1][0])**2+(center[0][1]-center[1][1])**2+(center[0][2]-center[1][2])**2+(center[0][3]-center[1][3])**2
    c2 = (center[0][0]-center[2][0])**2+(center[0][1]-center[2][1])**2+(center[0][2]-center[2][2])**2+(center[0][3]-center[2][3])**2
    c3 = (center[1][0]-center[2][0])**2+(center[1][1]-center[2][1])**2+(center[1][2]-center[2][2])**2+(center[1][3]-center[2][3])**2
    c_sum = c1+c2+c3
    dis = cd_total#+c_sum
    print("enddddddd:",dis)
    return 1/dis

def which_cls(center,X,cls_num):
    cd_total = 0
    c_d = []
    item_cls = []
    cd_min = 0
    for m in X:
        for i in range(cls_num):
            c_d.append((m[0]-center[i][0])**2+(m[1]-center[i][1])**2+(m[2]-center[i][2])**2+(m[3]-center[i][3])**2)
        cd_min = min(c_d)
        cls = c_d.index(cd_min)
        item_cls.append(cls)
        c_d = []
    return item_cls
            



def acc(Y):
    cont = 0
    for i in range(150):
        if y[i]!=Y[i]:
            cont+=1
    Acc = 1-(cont/150)
    return Acc

fitness_best = []


#种群各类参数

N=50                

Genome=96            

generation_max=100    

                     




av_fitness = []

popSize=N+1

genomeLength=Genome+1

top_bottom=3

QuBitZero = np.array([[1],[0]])

QuBitOne = np.array([[0],[1]])

AlphaBeta = np.empty([top_bottom])

fitness = np.empty([popSize])

probability = np.empty([popSize])

# qpv: quantum chromosome (or population vector, QPV)

qpv = np.empty([popSize, genomeLength, top_bottom])         

nqpv = np.empty([popSize, genomeLength, top_bottom])

# chromosome: classical chromosome

chromosome = np.empty([popSize, genomeLength],dtype=np.int) 

child1 = np.empty([popSize, genomeLength, top_bottom])

child2 = np.empty([popSize, genomeLength, top_bottom])

best_chrom = np.empty([generation_max])



#初始化全局变量

theta=0;

iteration=0;

the_best_chrom=0;

generation=0;




def Init_population():

    # Hadamard gate

    r2=math.sqrt(2.0)

    h=np.array([[1/r2,1/r2],[1/r2,-1/r2]])

    # Rotation Q-gate

    theta=0;

    rot =np.empty([2,2])

    # Initial population array (individual x chromosome)

    i=1; j=1;

    for i in range(1,popSize):

        for j in range(1,genomeLength):

            theta=np.random.uniform(0,1)*90

            theta=math.radians(theta)

            rot[0,0]=math.cos(theta); rot[0,1]=-math.sin(theta);

            rot[1,0]=math.sin(theta); rot[1,1]=math.cos(theta);

            AlphaBeta[0]=rot[0,0]*(h[0][0]*QuBitZero[0])+rot[0,1]*(h[0][1]*QuBitZero[1])

            AlphaBeta[1]=rot[1,0]*(h[1][0]*QuBitZero[0])+rot[1,1]*(h[1][1]*QuBitZero[1])

        # alpha squared          

            qpv[i,j,0]=np.around(2*pow(AlphaBeta[0],2),2) 
           

        # beta squared

            qpv[i,j,1]=np.around(2*pow(AlphaBeta[1],2),2) 





def Show_population():

    i=1; j=1;

    for i in range(1,popSize):

        print()

        print()

        print("qpv = ",i," : ")

        print()

        for j in range(1,genomeLength):

            print(qpv[i, j, 0],end="")

            print(" ",end="")

        print()

        for j in range(1,genomeLength):

            print(qpv[i, j, 1],end="")

            print(" ",end="")

    print()

    

#Obverse观测函数，坍塌阈值p_alpha

def Measure(p_alpha):#Obverse

    for i in range(1,popSize):

        #print()

        for j in range(1,genomeLength):

            if p_alpha<=qpv[i, j, 0]:

                chromosome[i,j]=0

            else:

                chromosome[i,j]=1

            #print(chromosome[i,j]," ",end="")

        #print()

    #print()



#适应度计算函数

def Fitness_evaluation(generation):

    i=1; j=1; fitness_total=0; sum_sqr=0;

    fitness_average=0; variance=0;
    x=0
    center=[[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    center_all = []
    global center_best


    for i in range(1,popSize):

        fitness[i]=0



#将二进制染色体串转化为十进制数

    for i in range(1,popSize):
        center=[[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        for k in range(3):#聚类中心数
            for m in range(4):#维数
                for n in range(1,9):#一个数值的位数
                    x=x+chromosome[i][k*32+m*8+n]*pow(2,n-1)
                center[k][m] = (x/pow(2,8))*x_p[m]+x_min[m]
                x=0
                
        
        f = distance(center,X,cls_num)

        #fitness[i]=y*100
        fitness[i] = f
        center_all.append(center)
        #print("...............................")
        #print(center_all)


      

        #print("fitness = ",f," ",fitness[i])

        fitness_total=fitness_total+fitness[i]

    fitness_average=fitness_total/N
    av_fitness.append(fitness_average)

    i=1;

    the_best_chrom=0;

    fitness_max=fitness[1];

    for i in range(1,popSize):

        if fitness[i]>=fitness_max:

            fitness_max=fitness[i]

            the_best_chrom=i

    best_chrom[generation]=the_best_chrom
    fitness_best.append(fitness[the_best_chrom])
    center_best = center_all[the_best_chrom-1]
    print("...............................")
    print("best_i:",the_best_chrom-1)
    #print("************************")
    #print(center_best)

    # Statistical output

    #f = open("output.dat","a")

    #f.write(str(generation)+" "+str(fitness_average)+"\n")

    #f.write(" \n")

    #f.close()

    print("Population size = ",popSize - 1)

    print("mean fitness = ",fitness_average)
    
    print("fitness max = ",fitness[the_best_chrom])

    #print("fitness sum = ",fitness_total)



#量子旋转门更新

def rotation():

    rot=np.empty([2,2])

    # Lookup table of the rotation angle

    for i in range(1,popSize):

        for j in range(1,genomeLength):
            #print("---------here-----------")
            #print(fitness[i])
            #print(best_chrom[generation])
            #b_g = int(best_chrom[generation])
            #best_chrom[generation] = b_g
            #print(best_chrom[generation])
            #print(fitness[best_chrom[generation]])
            #print("---------here-----------")

            if fitness[i]<fitness[int(best_chrom[generation])]:

# if chromosome[i,j]==0 and chromosome[best_chrom[generation],j]==0:

                if chromosome[i,j]==0 and chromosome[int(best_chrom[generation]),j]==1:

                    #1.固定值：旋转角0.03pi

                    #delta_theta=0.0942477796
                    
                    #2.随进化代数变化
                    delta_theta = 0.1*pi-(0.1*pi-0.005*pi)*generation/generation_max


                    rot[0,0]=math.cos(delta_theta); rot[0,1]=-math.sin(delta_theta);

                    rot[1,0]=math.sin(delta_theta); rot[1,1]=math.cos(delta_theta);

                    nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])

                    nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])

                    qpv[i,j,0]=round(nqpv[i,j,0],2)

                    qpv[i,j,1]=round(1-nqpv[i,j,0],2)

                if chromosome[i,j]==1 and chromosome[int(best_chrom[generation]),j]==0:



                    #delta_theta=-0.0942477796
                    #2.随进化代数变化
                    delta_theta = -(0.1*pi-(0.1*pi-0.005*pi)*generation/generation_max)

                    rot[0,0]=math.cos(delta_theta); rot[0,1]=-math.sin(delta_theta);

                    rot[1,0]=math.sin(delta_theta); rot[1,1]=math.cos(delta_theta);

                    nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])

                    nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])

                    qpv[i,j,0]=round(nqpv[i,j,0],2)

                    qpv[i,j,1]=round(1-nqpv[i,j,0],2)

             # if chromosome[i,j]==1 and chromosome[best_chrom[generation],j]==1:




#量子染色体突变

def mutation(pop_mutation_rate, mutation_rate):

    

    for i in range(1,popSize):

        up=np.random.random_integers(100)

        up=up/100

        if up<=pop_mutation_rate:

            for j in range(1,genomeLength):

                um=np.random.random_integers(100)

                um=um/100

                if um<=mutation_rate:

                    nqpv[i,j,0]=qpv[i,j,1]

                    nqpv[i,j,1]=qpv[i,j,0]

                else:

                    nqpv[i,j,0]=qpv[i,j,0]

                    nqpv[i,j,1]=qpv[i,j,1]

        else:

            for j in range(1,genomeLength):

                nqpv[i,j,0]=qpv[i,j,0]

                nqpv[i,j,1]=qpv[i,j,1]

    for i in range(1,popSize):

        for j in range(1,genomeLength):

            qpv[i,j,0]=nqpv[i,j,0]

            qpv[i,j,1]=nqpv[i,j,1]

            
            
#全干扰交叉
def all_cross():
    qpv_new = qpv
    qpv_a = qpv
    qpv_swap = qpv[1,1]
    print(qpv[2,2])

    for i in range(1,popSize):
        for j in range(1,genomeLength):
            if j >=2:
                k = i-j+1
                if i-j+1 <=0:#当需交换的下标为0或小于等于0时再向前一步
                    k=k-1
                qpv_swap = qpv_a[i,k]
                qpv_new[i,j] = qpv_swap
                qpv[i,j] = qpv_new[i,j]
    print(qpv[2,2])



#画图
def plot_Output():

    #data = np.loadtxt('output.dat')

    # plot the first column as x, and second column as y

    x=range(0,generation_max)

    y=fitness_best

    plt.plot(x,y)

    plt.xlabel('Generation')

    plt.ylabel('Fitness best')

    plt.xlim(0.0, 100.0)

    plt.show()



#QGA主程序

def Q_GA():
    

    generation=0;

    print("============== GENERATION: ",generation," =========================== ")

    print()

    Init_population()

    #Show_population()

    Measure(0.5)

    Fitness_evaluation(generation)

    while (generation<generation_max-1):

        print()

        print("============== GENERATION: ",generation+1," =========================== ")

        print()

        rotation()

        mutation(0.01,0.002)
        
        #if(generation>20 and fitness_best[generation]==fitness_best[generation-10]):
            #a=1
            #all_cross()
        #print(chromosome)
        #print("zhixing!!")
            
        Measure(0.5)

        Fitness_evaluation(generation)
        print("The best of generation [",generation,"] ", best_chrom[generation])
        generation=generation+1
        print("!!!!!!!!--------------")
        print(center_best)
       
        
        




df = pd.read_csv("Iris.csv",header=None)
y = df.iloc[0:,5].values
X = df.iloc[0:,1:5].values
cls_num = 3
center_best=[[0,0,0,0],[0,0,0,0],[0,0,0,0]]
center=[[0,0,0,0],[0,0,0,0],[0,0,0,0]]
x_p=[]
x_min=[]
y_cls=[]
x1 = df.iloc[0:,1].values
x_min.append(min(x1))
x_p.append(max(x1)-min(x1))
x2 = df.iloc[0:,2].values
x_min.append(min(x2))
x_p.append(max(x2)-min(x2))
x3 = df.iloc[0:,3].values
x_min.append(min(x3))
x_p.append(max(x3)-min(x3))
x4 = df.iloc[0:,4].values
x_min.append(min(x4))
x_p.append(max(x4)-min(x4))

#随机初始一个center
for i in range(3):
    for j in range(4):
        a = rd.random()
        center[i][j] = (x_min[j]+x_p[j]*a)
print("初始化center",center)


Q_GA()
print("---------------------------------------")
print("x_p:",x_p)
print("x_min:",x_min)
#print(center)
print("********************")
print(center_best)
y_cls = which_cls(center_best,X,cls_num)
print(y_cls)
A_num = 0
B_num = 0
C_num = 0
for i in range(150):
    if y_cls[i]==0:
        A_num+=1
    if y_cls[i]==1:
        B_num+=1
    if y_cls[i]==2:
        C_num+=1
print("第一类有:",A_num,"个")
print("第二类有:",B_num,"个")
print("第三类有:",C_num,"个")

f_count=0
if A_num<50:
    f_count+=50-A_num
if B_num<50:
    f_count+=50-B_num
if C_num<50:
    f_count+=50-C_num
    
print("Acc：",1-f_count/150)
#plot_Output()
