import pandas as pd
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import svm
import sklearn
import random
import tensorflow as tf
from sklearn import preprocessing
import pickle
from math import log
from sklearn.neural_network import MLPClassifier
import data_02_05 as d25

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

mean1 = []
for i in range(2):
    mean1.append(1.5)

mean2 = []
for i in range(2):
    mean2.append(-1.5)
    
cov = np.eye(2,2)
Gauss_class1 = np.random.multivariate_normal(mean1, cov, 200)
Gauss_class2 = np.random.multivariate_normal(mean2, cov, 200)
label1 = np.zeros((200,1))
label2 = []
for i in label1:
    label2.append(i+1)
    
Gaussa = np.hstack((Gauss_class1,label1))
Gaussb = np.hstack((Gauss_class2,label2))
Gauss = np.vstack((Gaussa,Gaussb))



# Gauss3:2-class,2-dim,400-sample
Gauss = pd.DataFrame(Gauss)
Gauss.iloc[:,:-1] = preprocessing.scale(Gauss.iloc[:,:-1])

iris = pd.read_csv(r'E:\机器学习任务\Iris.txt',names=list('abcde'))
for i in range(150):
    if iris.iloc[i,-1] == 'Iris-setosa':
        iris.iloc[i,-1] = 0
    elif iris.iloc[i,-1] == 'Iris-virginica':
        iris.iloc[i,-1] = 2
    else: iris.iloc[i,-1] = 1
# iris:3-class,4-dim,150-sample
iris.iloc[6,-1] = 0
iris.iloc[:,:-1] = preprocessing.scale(iris.iloc[:,:-1])

# seeds:3-class,7-dim,210-sample,70-70-70
seeds = pd.read_excel(r'E:\机器学习任务\seeds.xlsx',names=list('abcdefgh'))
seeds.iloc[:,:-1] = preprocessing.scale(seeds.iloc[:,:-1])

# haberman: 2-class,3-dim,306-sample,225个类1，81个类2
haberman = pd.read_excel(r'E:\机器学习任务\haberman.xlsx',names=list('abcd'))
##haberman.iloc[:,:-1] = preprocessing.scale(haberman.iloc[:,:-1])

# DUMDHTK:4-class,5-dim,258-sample,24 for verylow,83 for low,88 for middle,63 for high  
##DUMDHTK = pd.read_excel(r'E:\机器学习任务\Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN1.xlsx',names=list('abcdef'))
##globle_m = len(DUMDHTK)
##for i in range(globle_m):
##    if DUMDHTK.iloc[i,-1] == 'very_low':
##        DUMDHTK.iloc[i,-1] = 0
##    elif DUMDHTK.iloc[i,-1] == 'Low':
##        DUMDHTK.iloc[i,-1] = 1
##    elif DUMDHTK.iloc[i,-1] == 'Middle':
##        DUMDHTK.iloc[i,-1] = 2
##    else:
##        DUMDHTK.iloc[i,-1] = 3
##DUMDHTK.iloc[:,:-1] = preprocessing.scale(DUMDHTK.iloc[:,:-1])

# bannote authentication:2-class,4-dim,1372-sample,762 class1,610 class2
bannote = pd.read_excel(r'E:\机器学习任务\bannote authentication.xlsx',names=list('abcde'))
bannote.iloc[:,:-1] = preprocessing.scale(bannote.iloc[:,:-1])

# pima:2-class,8-dim,768-sample
pima = pd.read_excel(r'C:\Users\Administrator\Desktop\UCI数据\pima.xlsx',names=list('abcdefghi'))
pima.iloc[:,:-1] = preprocessing.scale(pima.iloc[:,:-1])

# vertebral:3-class,6-dim,310-sample,60 class1,150 class2,100 class3
##vertebral = pd.read_excel(r'C:\Users\Administrator\Desktop\UCI数据\vertebral.xlsx',names=list('abcdefg'))
##vertebral.iloc[:,:-1] = preprocessing.scale(vertebral.iloc[:,:-1])
##for i in range(len(vertebral)):
##    if vertebral.iloc[i,-1] == 'DH':
##        vertebral.iloc[i,-1] = 0
##    elif vertebral.iloc[i,-1] == 'SL':
##        vertebral.iloc[i,-1] = 1
##    elif vertebral.iloc[i,-1] == 'NO':
##        vertebral.iloc[i,-1] = 2
      
# wine:3-class,13-dim,178-sample,59 class1,71 class2,48 class3
wine = pd.read_excel(r'C:\Users\Administrator\Desktop\UCI数据\wine.xlsx',names=list('abcdefghijklop'))
wine.iloc[:,:-1] = preprocessing.scale(wine.iloc[:,:-1])

# cirdata:2-class,2-dim,400-sample,200-class1,200-class2
##cirdata = d25.get_cirdata()
##cirdata[:,:-1] = preprocessing.scale(cirdata[:,:-1])

def distence(x,y):
    dis = np.sqrt(sum((x-y)**2))
    return dis



def density(data,U_Ind,D):
    m = np.shape(data)[0]
    denList = []
    dc = Get_dc(data,D)
    for i in U_Ind:
        SortDisList = Get_DisList(i,data)
        for j in range(m):
            if (SortDisList[j] <= dc) and (SortDisList[j+1] > dc):
                denList.append(j)
                break
    return denList
        


# D表示dc与数据宽度的比例
def Get_dc(data,D):
    m = np.shape(data)[0]
    A = []
    data = np.array(data)
    for i in range(m):
        curDisList = []
        for j in range(m):
            curDisList.append(distence(data[i,:-1],data[j,:-1]))
        A.append(sorted(curDisList))

    A = np.array(A)   
    dc = sorted(A[:,-1])[-1] * D
    return dc




def Get_DisList(samIndex,data,sortdata=True):
    m = np.shape(data)[0]
    data = np.array(data)
    disList = []
    for i in range(m):
        disList.append(distence(data[samIndex,:-1],data[i,:-1]))
    if sortdata==True:
        return sorted(disList)
    elif sortdata==False:
        return disList




        
# N表示有标记样本的比例
# 此函数也可以用于从无标记数据分出测试集
# 该函数只适用于分为三类，且每一类数目相当的数据集
def SplitSet(data,N):
    m = np.shape(data)[0]
    allInd = list(range(m))
    if N<=0.5:
        labInd = range(0,m,int(1/N))

    else:
##        labInd = random.sample(allInd,int(m*N))
        
        labInd = []
        labNum = int(m * N)
        labNum3 = int((m * N) / 3)
        labInd.extend(random.sample(range(int(m/3)),labNum3))
        labInd.extend(random.sample(range(int(m/3),int(2*m/3)),labNum3))
        labInd.extend(random.sample(range(int(2*m/3),m),(labNum-(2*labNum3))))
        
    for i in labInd:
        allInd.remove(i)
    labset = []
    unlabset = []
    for j in labInd:
        labset.append(data[j])

    for k in allInd:
        unlabset.append(data[k])

            
    return np.array(labset,dtype='float32'),np.array(unlabset,dtype='float32')


# 返回L和U的index
def SplitSet_Ind(data_Ind,N):
    m = len(data_Ind)
    allInd = list(data_Ind)
    labInd = []
##    for i in data_Ind:
##        if i%(int(m/(m*N))) == 0:
##            labInd.append(i)
    
    labInd.extend(random.sample(allInd,int(m*N)))

    
    for i in labInd:
        allInd.remove(i)  

    return np.array(labInd,dtype=np.int32),np.array(allInd,dtype=np.int32)
    

def StaKNN(data,M,N,K):
    knn = neighbors.KNeighborsClassifier(n_neighbors=K)
    L,U0 = SplitSet(np.array(data),M)
    U,T = SplitSet(U0,N)
    knn.fit(L[:,:-1],L[:,-1])

    errcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(knn.predict(T[i,:-1].reshape(1,-1))):
            errcount += 1
    return (errcount/len(T))





# 画出随M增大的20次结果的平均图像
def Graph_M(classifier,data,N,K):
    count = 0
    errlist0 = [0]*9
    errlist1 = []
    while (count<20):
        for i in np.arange(0.1,1,0.1):
            errlist1.append(classifier(data,i,N,K))
        errlist = list(map(lambda x:x[0]+x[1],zip(errlist0,errlist1)))
        errlist0 = errlist
        errlist1 = []
        count += 1
    errlist = [i/20 for i in errlist0]

    plt.plot(np.arange(0.1,1,0.1),errlist,'-x')
    plt.show()



# 每次从U中随机取一个数据对knn进行训练的自训练算法
def random_self_KNN(data,L,U,T,K):
    knn = neighbors.KNeighborsClassifier(n_neighbors=K)
    knn.fit(L[:,:-1],L[:,-1])
    errlist = []
    random.shuffle(U)
##    U = [146, 112, 113, 132, 104, 121, 130, 139, 2, 3, 6, 10, 11, 18, 21, 22, \
##         24, 25, 26, 27, 29, 30, 31, 33, 34, 39, 42, 43, 52, 58, 61, 62, 63, \
##         67, 68, 73, 77, 80, 81, 83, 84, 85, 87, \
##         92, 94, 95, 97, 99, 110, 111, 116, 122, 126, 127, 133, 135, 138, \
##         143, 144, 148]
    U = get_data_from_index(data,U)
    for j in range(len(U)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(knn.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U[j,-1] = int(knn.predict(U[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U[j]))
        knn.fit(L[:,:-1],L[:,-1])
    lastcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(knn.predict(T[i,:-1].reshape(1,-1))):
            lastcount += 1
    errlist.append(lastcount)


    return errlist

# 每次从U中随机取一个数据对SVM进行训练的自训练算法   
def random_self_SVM(data,L,U,T):
    clf = svm.SVC()
    clf.fit(L[:,:-1],L[:,-1])
    errlist = []
    random.shuffle(U)
    U = get_data_from_index(data,U)
    for j in range(len(U)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U[j,-1] = int(clf.predict(U[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U[j]))
        clf.fit(L[:,:-1],L[:,-1])
    lastcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            lastcount += 1
    errlist.append(lastcount)
    return errlist

def random_self_SLNN(data,L,U,T):
    clf = MLPClassifier(hidden_layer_sizes=(64),max_iter=2000,\
                        learning_rate_init=0.005,shuffle=False)
    clf.fit(L[:,:-1],L[:,-1])
    errlist = []
    random.shuffle(U)
    U = get_data_from_index(data,U)
    for j in range(len(U)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U[j,-1] = int(clf.predict(U[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U[j]))
        clf.fit(L[:,:-1],L[:,-1])
    lastcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            lastcount += 1
    errlist.append(lastcount)
    return errlist



def all_self_SVM(data,L,U,T):
    clf = svm.SVC()
    clf.fit(L[:,:-1],L[:,-1])
    U = get_data_from_index(data,U)
    errlist = []
    errcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            errcount += 1
    errlist.append(errcount)

    for j in range(len(U)):
        U[j,-1] = int(clf.predict(U[j,:-1].reshape(1,-1)))
    L = np.vstack((L,U))
    clf.fit(L[:,:-1],L[:,-1])
    errcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            errcount += 1
    errlist.append(errcount)
    return errlist


def all_self_KNN(data,L,U,T,K):
    clf = neighbors.KNeighborsClassifier(n_neighbors=K)
    clf.fit(L[:,:-1],L[:,-1])
    U = get_data_from_index(data,U)
    errlist = []
    errcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            errcount += 1
    errlist.append(errcount)

    for j in range(len(U)):
        U[j,-1] = int(clf.predict(U[j,:-1].reshape(1,-1)))
    L = np.vstack((L,U))
    clf.fit(L[:,:-1],L[:,-1])
    errcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            errcount += 1
    errlist.append(errcount)
    return errlist

def all_self_SLNN(data,L,U,T):
    clf = MLPClassifier(hidden_layer_sizes=(64),max_iter=2000,\
                        learning_rate_init=0.005,shuffle=False)
    clf.fit(L[:,:-1],L[:,-1])
    U = get_data_from_index(data,U)
    errlist = []
    errcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            errcount += 1
    errlist.append(errcount)

    for j in range(len(U)):
        U[j,-1] = int(clf.predict(U[j,:-1].reshape(1,-1)))
    L = np.vstack((L,U))
    clf.fit(L[:,:-1],L[:,-1])
    errcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            errcount += 1
    errlist.append(errcount)
    return errlist



# 画出随M比例变大SVM的20次平均图像
def Graph_M_SVM(data,N):
    count = 0
    errlist0 = [0]*9
    errlist1 = []
    while (count<20):
        for i in np.arange(0.1,1,0.1):
            errlist1.append(StaSVM(data,i,N))
        errlist = list(map(lambda x:x[0]+x[1],zip(errlist0,errlist1)))
        errlist0 = errlist
        errlist1 = []
        count += 1
    errlist = [i/20 for i in errlist0]

    plt.plot(np.arange(0.1,1,0.1),errlist,'-x')
    plt.show()




# 测试算法的波动
def stability_test(data,M):
    num = 0
    relist = []
    while num<20:
        l,t = SplitSet(np.array(data),M)
    ##    clf = neighbors.KNeighborsClassifier(n_neighbors=4)
        clf = svm.SVC()
        clf.fit(l[:,:-1],l[:,-1])
        errcount = 0
        for i in range(len(t)):
                if t[i,-1] != int(clf.predict(t[i,:-1].reshape(1,-1))):
                        errcount += 1
        relist.append(errcount)
        num += 1
    plt.plot(relist,'-x')
    plt.ylabel('error rate')
    plt.show()


# 返回值是按U中密度从大到小排列的索引
def sort_data_ind(data,U_Ind,D):
    data = np.array(data)
    denlist = density(data,U_Ind,D)
    return list(reversed(np.argsort(denlist)))



def get_data_from_index(data,indlist):
    data = np.array(data)
    outdata = []
    for i in indlist:
        outdata.append(data[i])
    return np.array(outdata,dtype=np.float32)


# 传入参数中L,T是具体的数据集，而U是索引的集合    
def self_SVM(data,L,U,T,D):
    clf = svm.SVC() 
    clf.fit(L[:,:-1],L[:,-1])
    errlist = []
    U_data = get_data_from_index(data,U)
    sortedU = []
    sortU_ind = sort_data_ind(data,U,D)
    for i in sortU_ind:
        sortedU.append(U_data[i])
    U = sortedU
    U = np.array(sortedU,dtype=np.int32)    
    for j in range(len(U)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U[j,-1] = int(clf.predict(U[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U[j]))
        clf.fit(L[:,:-1],L[:,-1])
    lastcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            lastcount += 1
    errlist.append(lastcount)
##    print(errlist)
    
    return errlist
    
def self_KNN(data,L,U,T,D,K):
    clf = neighbors.KNeighborsClassifier(n_neighbors=K)
    clf.fit(L[:,:-1],L[:,-1])
    errlist = []
    U_data = get_data_from_index(data,U)
    sortedU = []
    sortU_ind = sort_data_ind(data,U,D)
    for i in sortU_ind:
        sortedU.append(U_data[i])
    U = np.array(sortedU,dtype=np.int32)
    for j in range(len(U)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U[j,-1] = int(clf.predict(U[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U[j]))
        clf.fit(L[:,:-1],L[:,-1])
    lastcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            lastcount += 1
    errlist.append(lastcount)        
##    print(errlist)    
    return errlist

def self_SLNN(data,L,U,T,D):
    clf = MLPClassifier(hidden_layer_sizes=(64),max_iter=2000,\
                        learning_rate_init=0.005,shuffle=False)
    clf.fit(L[:,:-1],L[:,-1])
    errlist = []
    U_data = get_data_from_index(data,U)
    sortedU = []
    sortU_ind = sort_data_ind(data,U,D)
    for i in sortU_ind:
        sortedU.append(U_data[i])
    U = np.array(sortedU,dtype=np.int32)
    for j in range(len(U)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U[j,-1] = int(clf.predict(U[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U[j]))
        clf.fit(L[:,:-1],L[:,-1])
    lastcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            lastcount += 1
    errlist.append(lastcount)           
    return errlist

    

def greedy(data,L,U,T):
    L = get_data_from_index(data,L)
    T = get_data_from_index(data,T)
##    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    clf = svm.SVC()
    clf.fit(L[:,:-1],L[:,-1])
    U_data = get_data_from_index(data,U)
    bestInd = []
    errlist = []
    while (len(U_data) != 0):
        err = float('inf')
        bestoneall = 0
        bestone = 0
        for i in range(len(U_data)):
            U_data[i,-1] = int(clf.predict(U_data[i,:-1].reshape(1,-1)))
            L0 = np.vstack((L,U_data[i]))
            clf.fit(L0[:,:-1],L0[:,-1])
            curerr = 0
            for j in range(60):
                if T[j,-1] != int(clf.predict(T[j,:-1].reshape(1,-1))):
                    curerr += 1
            if curerr < err:
                err = curerr
                bestone = i # U_data的索引
        bestInd.append(U[bestone]) # U[bestone]是data的索引
        errlist.append(err)
        U_data = np.delete(U_data,bestone,0)
        U = np.delete(U,bestone,0)
    return errlist,bestInd


##f = open(r'E:\机器学习任务\iris_LUT\L.txt','rb')
##L = pickle.load(f)
##f.close()
##f = open(r'E:\机器学习任务\iris_LUT\U.txt','rb')
##U = pickle.load(f)
##f.close()
##f = open(r'E:\机器学习任务\iris_LUT\T.txt','rb')
##T = pickle.load(f)
##f.close()

# R表示核心点core占总样本数量的比例
def CBNindex(data,D,R):
    m = np.shape(data)[0]
    allindex = list(range(m))
    denlist = density(data,allindex,D)
    sortdenlist = sorted(denlist,reverse=True)
    denThreshold = sortdenlist[int(len(denlist)*R)]
    denlist = np.array(denlist)
    allindex0 = np.array(allindex)
    coreindex = allindex0[denlist>=denThreshold]
    dc = Get_dc(data,D)
    for i in coreindex:
        allindex.remove(i)
    boundindex = []
    for i in coreindex:
        dislist = Get_DisList(i,data,sortdata=False)
        for j in range(len(dislist)):
            if (dislist[j] < dc) and (j not in coreindex):
                boundindex.append(j)
                continue

    boundindex = list(set(boundindex))
    for i in boundindex:
        allindex.remove(i)

    noiseindex = allindex
    return coreindex,boundindex,noiseindex
            
def plot_CBN(data,D,R):   
    C,B,N = CBNindex(data,D,R)
    C = get_data_from_index(data,C)
    B = get_data_from_index(data,B)
    N = get_data_from_index(data,N)
    plt.scatter(C[:,0],C[:,1],marker='s',label='核心点')
    plt.scatter(B[:,0],B[:,1],marker='+',label='边界点')
    plt.scatter(N[:,0],N[:,1],marker='o',label='噪声点')
    plt.legend()
    plt.show()

def D_self_KNN(data,L,U,T,D,K,R):
    clf = neighbors.KNeighborsClassifier(n_neighbors=K)
    clf.fit(L[:,:-1],L[:,-1])
    C,B,N = CBNindex(data,D,R)
    U_C = []
    U_B = []
    U_N = []
    for i in U:
        if i in C:
            U_C.append(i)
        elif i in B:
            U_B.append(i)
        elif i in N:
            U_N.append(i)
    U_C = get_data_from_index(data,U_C)
    U_B = get_data_from_index(data,U_B)
    U_N = get_data_from_index(data,U_N)

    errlist = []
    for j in range(len(U_C)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U_C[j,-1] = int(clf.predict(U_C[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U_C[j]))
        clf.fit(L[:,:-1],L[:,-1])
    
    for j in range(len(U_B)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U_B[j,-1] = int(clf.predict(U_B[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U_B[j]))
        clf.fit(L[:,:-1],L[:,-1])

    lastcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            lastcount += 1
    errlist.append(lastcount)
        
    return errlist

def D_self_SVM(data,L,U,T,D,R):
    clf = svm.SVC()
    clf.fit(L[:,:-1],L[:,-1])
    C,B,N = CBNindex(data,D,R)
    U_C = []
    U_B = []
    U_N = []
    for i in U:
        if i in C:
            U_C.append(i)
        elif i in B:
            U_B.append(i)
        elif i in N:
            U_N.append(i)
    U_C = get_data_from_index(data,U_C)
    U_B = get_data_from_index(data,U_B)
    U_N = get_data_from_index(data,U_N)

    errlist = []
    for j in range(len(U_C)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U_C[j,-1] = int(clf.predict(U_C[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U_C[j]))
        clf.fit(L[:,:-1],L[:,-1])
    
    for j in range(len(U_B)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U_B[j,-1] = int(clf.predict(U_B[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U_B[j]))
        clf.fit(L[:,:-1],L[:,-1])

    lastcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            lastcount += 1
    errlist.append(lastcount)
        
    return errlist    

def D_self_SLNN(data,L,U,T,D,R):
    clf = MLPClassifier(hidden_layer_sizes=(64),max_iter=2000,\
                        learning_rate_init=0.005,shuffle=False)
    clf.fit(L[:,:-1],L[:,-1])
    C,B,N = CBNindex(data,D,R)
    U_C = []
    U_B = []
    U_N = []
    for i in U:
        if i in C:
            U_C.append(i)
        elif i in B:
            U_B.append(i)
        elif i in N:
            U_N.append(i)
    U_C = get_data_from_index(data,U_C)
    U_B = get_data_from_index(data,U_B)
    U_N = get_data_from_index(data,U_N)

    errlist = []
    for j in range(len(U_C)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U_C[j,-1] = int(clf.predict(U_C[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U_C[j]))
        clf.fit(L[:,:-1],L[:,-1])
    
    for j in range(len(U_B)):
        errcount = 0
        for i in range(len(T)):
            if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
                errcount += 1
        errlist.append(errcount)
        U_B[j,-1] = int(clf.predict(U_B[j,:-1].reshape(1,-1)))
        L = np.vstack((L,U_B[j]))
        clf.fit(L[:,:-1],L[:,-1])

    lastcount = 0
    for i in range(len(T)):
        if T[i,-1] != int(clf.predict(T[i,:-1].reshape(1,-1))):
            lastcount += 1
    errlist.append(lastcount)
        
    return errlist


#  注意class的数目与数据相符合 
def find_cen(L):
    m = len(L)
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    for i in range(m):
        if L[i,-1] == 0:
            class1.append(L[i])
        elif L[i,-1] == 1:
            class2.append(L[i])
        elif L[i,-1] == 2:
            class3.append(L[i])
##        elif L[i,-1] == 3:
##            class4.append(L[i])
    cen1 = np.sum(class1,axis=0)/np.array(len(class1))
    cen2 = np.sum(class2,axis=0)/np.array(len(class2))
    cen3 = np.sum(class3,axis=0)/np.array(len(class3))
##    cen4 = np.sum(class4,axis=0)/np.array(len(class4))
    return [cen1,cen2,cen3],[np.array(class1),np.array(class2)]



##plt.scatter(classlist[0][:,0],classlist[0][:,1],c='r',marker='o',label='class1')
##plt.scatter(classlist[1][:,0],classlist[1][:,1],c='b',marker='o',label='class2')
##plt.scatter(classlist[2][:,0],classlist[2][:,1],c='g',marker='o',label='class3')
##plt.scatter(cen[0][0],cen[0][1],c='r',marker='*',s=200,label='cen1')
##plt.scatter(cen[1][0],cen[1][1],c='b',marker='*',s=200,label='cen2')
##plt.scatter(cen[2][0],cen[2][1],c='g',marker='*',s=200,label='cen3')
##plt.legend()
##plt.show()

def get_entropy(problist):
    m = len(problist)
    Ent = 0
    for i in range(m):
        Ent -= problist[i] * log(problist[i],2)
    return Ent


def group_based_entropy(data,U_index,cen):
    U = get_data_from_index(data,U_index)
    U_dislist = []
    for i in range(len(U)):
        curlist = []
        for j in range(len(cen)):
            curlist.append(1/(distence(U[i, :-1],cen[j][:-1])))
        U_dislist.append(curlist)

    sumlist = []
    for i in range(len(U)):
        sumlist.append(sum(U_dislist[i]))

    for i in range(len(U)):
        U_dislist[i] = np.array(U_dislist[i])/np.array(sumlist[i])

    entropy_list = []
    for i in range(len(U)):
        entropy_list.append(get_entropy(U_dislist[i]))

    return entropy_list

def plot_graph(data,D,K,R):
    L,U,T = d25.iris_02_05()
##    L,U,T = d25.bannote_02_05()
##    L,U,T = d25.haberman_02_05()
##    L,U,T = d25.gauss3_02_05()
##    L,U,T = d25.DUMDHTK_02_05()
##    L,U,T = d25.pima_02_05()
##    L,U,T = d25.seeds_02_05()
##    L,U,T = d25.vertebral_02_05()
##    L,U,T = d25.wine_02_05()
##    L,U,T = d25.cirdata_02_05()
    L = get_data_from_index(data,L)
    T = get_data_from_index(data,T)
##    cen,classlist = find_cen(L)
##
##    entropy = group_based_entropy(data,U,cen)
##    groupU = []
##    for i in range(len(U)):
##        if entropy[i] < Ent:
##            groupU.append(U[i])


    n = len(U)+1    
##    errlist_selfKNN = [0]*n
##    errlist_selfSVM = [0]*n
##    errlist_selfSLNN = [0]*n
    errlist_randomKNN = [0]*n
    errlist_randomSVM = [0]*n
    errlist_randomSLNN = [0]*n
##    errlist_allSVM = [0]*2
##    errlist_allKNN = [0]*2
##    errlist_allSLNN = [0]*2
    errlist_dselfKNN = [0]*n
    errlist_dselfSVM = [0]*n
    errlist_dselfSLNN = [0]*n
    
    count = 0
    while(count<10):
##        selfknn = self_KNN(data,L,U,T,D,K)
##        selfsvm = self_SVM(data,L,U,T,D)
##        selfslnn = self_SLNN(data,L,U,T,D)
        randomknn = random_self_KNN(data,L,U,T,K)
        randomsvm = random_self_SVM(data,L,U,T)
        randomslnn = random_self_SLNN(data,L,U,T)
##        allknn = all_self_KNN(data,L,U,T,K)
##        allsvm = all_self_SVM(data,L,U,T)
##        allslnn = all_self_SLNN(data,L,U,T)
        dselfknn = D_self_KNN(data,L,U,T,D,K,R)
        dselfsvm = D_self_SVM(data,L,U,T,D,R)
        dselfslnn = D_self_SLNN(data,L,U,T,D,R)
        
##        errlist_selfKNN = list(map(lambda x:x[0]+x[1],zip(errlist_selfKNN,selfknn)))
##        errlist_selfSVM = list(map(lambda x:x[0]+x[1],zip(errlist_selfSVM,selfsvm)))
##        errlist_selfSLNN = list(map(lambda x:x[0]+x[1],zip(errlist_selfSLNN,selfslnn)))
        errlist_randomKNN = list(map(lambda x:x[0]+x[1],zip(errlist_randomKNN,randomknn)))
        errlist_randomSVM = list(map(lambda x:x[0]+x[1],zip(errlist_randomSVM,randomsvm)))
        errlist_randomSLNN = list(map(lambda x:x[0]+x[1],zip(errlist_randomSLNN,randomslnn)))
##        errlist_allKNN = list(map(lambda x:x[0]+x[1],zip(errlist_allKNN,allknn)))
##        errlist_allSVM = list(map(lambda x:x[0]+x[1],zip(errlist_allSVM,allsvm)))
##        errlist_allSLNN = list(map(lambda x:x[0]+x[1],zip(errlist_allSLNN,allslnn)))
        errlist_dselfKNN = list(map(lambda x:x[0]+x[1],zip(errlist_dselfKNN,dselfknn)))
        errlist_dselfSVM = list(map(lambda x:x[0]+x[1],zip(errlist_dselfSVM,dselfsvm)))
        errlist_dselfSLNN = list(map(lambda x:x[0]+x[1],zip(errlist_dselfSLNN,dselfslnn)))
        
        count += 1
        L,U,T = d25.iris_02_05()
##        L,U,T = d25.bannote_02_05()
##        L,U,T = d25.haberman_02_05()
##        L,U,T = d25.gauss3_02_05()
##        L,U,T = d25.DUMDHTK_02_05()
##        L,U,T = d25.pima_02_05()
##        L,U,T = d25.seeds_02_05()
##        L,U,T = d25.vertebral_02_05()
##        L,U,T = d25.wine_02_05()
##        L,U,T = d25.cirdata_02_05()
        L = get_data_from_index(data,L)
        T = get_data_from_index(data,T)
        
##        cen,classlist = find_cen(L)
##
##        entropy = group_based_entropy(data,U,cen)
##        groupU = []
##        for i in range(len(U)):
##            if entropy[i] < Ent:
##                groupU.append(U[i])

    leng = len(T)
##    errlist_selfKNN = [(1-i/20/leng) for i in errlist_selfKNN]
##    errlist_selfSVM = [(1-i/20/leng) for i in errlist_selfSVM]
##    errlist_selfSLNN = [(1-i/20/leng) for i in errlist_selfSLNN]
    errlist_randomKNN = [(1-i/10/leng) for i in errlist_randomKNN]
    errlist_randomSVM = [(1-i/10/leng) for i in errlist_randomSVM]
    errlist_randomSLNN = [(1-i/10/leng) for i in errlist_randomSLNN]
##    errlist_allSVM = [(1-i/50/leng) for i in errlist_allSVM]
##    errlist_allKNN = [(1-i/50/leng) for i in errlist_allKNN]
##    errlist_allSLNN = [(1-i/50/leng) for i in errlist_allSLNN]
    errlist_dselfKNN = [(1-i/10/leng) for i in errlist_dselfKNN]
    errlist_dselfSVM = [(1-i/10/leng) for i in errlist_dselfSVM]
    errlist_dselfSLNN = [(1-i/10/leng) for i in errlist_dselfSLNN]
    
##    print('KNN:%f,SFKNN:%f,DSFKNN:%f'%(errlist_allKNN[0],errlist_allKNN[1],errlist_dselfKNN[-1]))
##    print('SVM:%f,SFSVM:%f,DSFSVM:%f'%(errlist_allSVM[0],errlist_allSVM[1],errlist_dselfSVM[-1]))
##    print('SLNN:%f,SFSLNN:%f,DSFSLNN:%f'%(errlist_allSLNN[0],errlist_allSLNN[1],errlist_dselfSLNN[-1]))
    
    fig = plt.figure
    ax1 = plt.subplot(131)
    plt.sca(ax1)
    plt.title(u'基于KNN算法')
    plt.ylabel(u'分类正确率')
    plt.xlabel(u'自训练迭代过程')
    plt.plot(errlist_dselfKNN,'r',label='OST-KNN')
    plt.plot(errlist_randomKNN,'b',label='ST-KNN')
    plt.legend()
    
    ax2 = plt.subplot(132)
    plt.sca(ax2)
    plt.title(u'基于SVM算法')
    plt.ylabel(u'分类正确率')
    plt.xlabel(u'自训练迭代过程')
    plt.plot(errlist_dselfSVM,'r',label='OST-SVM')
    plt.plot(errlist_randomSVM,'b',label='ST-SVM')
    plt.legend()

    ax3 = plt.subplot(133)
    plt.sca(ax3)
    plt.title(u'基于SLNN算法')
    plt.ylabel(u'分类正确率')
    plt.xlabel(u'自训练迭代过程')
    plt.plot(errlist_dselfSLNN,'r',label='OST-SLNN')
    plt.plot(errlist_randomSLNN,'b',label='ST-SLNN')
    plt.legend()

    plt.show()
    

##plot_graph(iris,0.1,5,0.5)


    
##L,U,T = d25.gauss3_02_05()
##L = get_data_from_index(Gauss,L)
##T = get_data_from_index(Gauss,T)
##cen,classlist = find_cen(L)
##entropy = group_based_entropy(Gauss,U,cen)
##groupU = []
##for i in range(len(U)):
##            if entropy[i] < 0.6:
##                groupU.append(U[i])
##
##groupU = get_data_from_index(Gauss,groupU)
##plt.scatter(Gauss.iloc[:200,0],Gauss.iloc[0:200,1],color='r',marker='o',label='class1')
##plt.scatter(Gauss.iloc[1000:2000,0],Gauss.iloc[1000:2000,1],color='b',marker='o',label='class2')
##plt.scatter(groupU[:,0],groupU[:,1],color='g',marker='+',label='goupU',s=100)
##plt.legend()
##plt.show()

##L,U,T = d25.bannote_02_05()
##L = get_data_from_index(bannote,L)
##cen,classlist = find_cen(L)
##entropy = group_based_entropy(bannote,U,cen)
##length = len(entropy)
##lengthind = int(length*0.8)
##entropy = sorted(entropy)
##Ent = entropy[lengthind]
##print(Ent)


    
## 将数据存为pickle
##C1,C2 = d25.testdata()
##fr = open(r'C:\Users\Administrator\Desktop\UCI数据\testdataC1.xlsx','wb')
##pickle.dump(C1,fr,True)
##fr.close()
##
##fr = open(r'C:\Users\Administrator\Desktop\UCI数据\testdataC2.xlsx','wb')
##pickle.dump(C2,fr,True)
##fr.close()

# 载入pickle数据
fr = open(r'C:\Users\Administrator\Desktop\UCI数据\testdataC1.xlsx','rb')
C1 = pickle.load(fr)
fr.close()
fr = open(r'C:\Users\Administrator\Desktop\UCI数据\testdataC2.xlsx','rb')
C2 = pickle.load(fr)
fr.close()

## 确定数据空间结构的图
fig = plt.figure
ax1 = plt.subplot(121)
plt.sca(ax1)
plt.xlim(0,10)
plt.ylim(0,10)
plt.scatter(C1[:,0],C1[:,1],color='black',marker='^',label=u'A类')
plt.scatter(C2[:,0],C2[:,1],color='black',marker='v',label=u'B类')
plt.legend()

ax2 = plt.subplot(122)
plt.sca(ax2)
Cind,Bind,Nind = CBNindex(C1,0.3,0.25)
C = get_data_from_index(C1,Cind)
B = get_data_from_index(C1,Bind)
N = get_data_from_index(C1,Nind)
plt.xlim(0,10)
plt.ylim(0,10)
plt.scatter(C[:,0],C[:,1],color='black',marker='o',label=u'核心点')
plt.scatter(B[:,0],B[:,1],color='black',marker='s',label=u'边界点')
plt.scatter(N[:,0],N[:,1],color='black',marker='x',label=u'噪声点')
C = sorted(C,key=lambda item: item[1],reverse=True)
C = np.array(C)
for i in range(len(C)-1):
    plt.plot(C[i:i+2,0],C[i:i+2,1],'k-',)

for i in B:
    dislist = []
    for j in C:
        dislist.append(distence(i,j))
    ind = np.argmin(dislist)
    plt.plot([i[0],C[ind,0]],[i[1],C[ind,1]],'k:')
    
    
    
    

##Cind,Bind,Nind = CBNindex(C2,0.3,0.25)
##C = get_data_from_index(C2,Cind)
##B = get_data_from_index(C2,Bind)
##N = get_data_from_index(C2,Nind)
##plt.scatter(C[:,0],C[:,1],color='black',marker='o')
##plt.scatter(B[:,0],B[:,1],color='black',marker='s')
##plt.scatter(N[:,0],N[:,1],color='black',marker='x')
##C = sorted(C,key=lambda item: item[1],reverse=True)
##C = np.array(C)
##for i in range(len(C)-1):
##    plt.plot(C[i:i+2,0],C[i:i+2,1],'k-',)
##
##for i in B:
##    dislist = []
##    for j in C:
##        dislist.append(distence(i,j))
##    ind = np.argmin(dislist)
##    plt.plot([i[0],C[ind,0]],[i[1],C[ind,1]],'k:')
##   
##plt.legend()
##plt.show()




##dislistB = []
##for i in B:
##    curlist = []
##    for j in C:
##        curlist.append(distence(i,j))
##    dislistB.append(sorted(curlist)[0])
##
##dislistN = []
##for i in N:
##    curlist = []
##    for j in C:
##        curlist.append(distence(i,j))
##    dislistN.append(sorted(curlist)[0])
##    
##print(dislistB)
##print(dislistN)

            
