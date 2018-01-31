import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

def iris_02_05():
    allInd = list(range(150))
    L = []
    U = []
    T = []
    L.extend(random.sample(allInd[0:50],4))
    L.extend(random.sample(allInd[50:100],4))
    L.extend(random.sample(allInd[100:150],4))
    for i in L:
        allInd.remove(i)
    U = allInd
    T.extend(random.sample(U[0:46],12))
    T.extend(random.sample(U[46:92],12))
    T.extend(random.sample(U[92:138],12))
    for i in T:
        U.remove(i)
    return L,U,T

def bannote_02_05():

    allInd = list(range(1371))
    L = []
    U = []
    T = []
    L.extend(random.sample(allInd[0:726],51))
    L.extend(random.sample(allInd[726:1371],52))
    for i in L:
        allInd.remove(i)
    U = allInd
    T.extend(random.sample(U[0:675],170))
    T.extend(random.sample(U[675:1268],160))
    for i in T:
        U.remove(i)
    return L,U,T

def haberman_02_05():
    allInd = list(range(305))
    L = []
    U = []
    T = []
    L.extend(random.sample(allInd[:],23))
    for i in L:
        allInd.remove(i)
    U = allInd
    T.extend(random.sample(U[:],70))
    for i in T:
        U.remove(i)
    return L,U,T

##def gauss3_02_05():
##    allInd = list(range(400))
##    L = []
##    U = []
##    T = []
##    L.extend(random.sample(allInd[0:200],40))
##    L.extend(random.sample(allInd[200:400],40))
##    for i in L:
##        allInd.remove(i)
##    U = allInd
##    T.extend(random.sample(U[0:160],80))
##    T.extend(random.sample(U[160:320],80))
##    for i in T:
##        U.remove(i)
##    return L,U,T

##def DUMDHTK_02_05():
##    allInd = list(range(257))
##    L = []
##    U = []
##    T = []
##    L.extend(random.sample(allInd[0:23],5))
##    L.extend(random.sample(allInd[23:111],17))
##    L.extend(random.sample(allInd[111:194],16))
##    L.extend(random.sample(allInd[194:257],12))
##    for i in L:
##        allInd.remove(i)
##    U = allInd
##    T.extend(random.sample(U[0:18],9))
##    T.extend(random.sample(U[18:90],36))
##    T.extend(random.sample(U[90:157],33))
##    T.extend(random.sample(U[157:207],25))
##    for i in T:
##        U.remove(i)
##    return L,U,T

def pima_02_05():
    allInd = list(range(767))
    L = []
    U = []
    T = []
    L.extend(random.sample(allInd[:],58))
    for i in L:
        allInd.remove(i)
    U = allInd
    T.extend(random.sample(U[:],177))
    for i in T:
        U.remove(i)
    return L,U,T



##def vertebral_02_05():
##    allInd = list(range(309))
##    L = []
##    U = []
##    T = []
##    L.extend(random.sample(allInd[0:60],12))
##    L.extend(random.sample(allInd[60:210],30))
##    L.extend(random.sample(allInd[210:309],20))
##    for i in L:
##        allInd.remove(i)
##    U = allInd
##    T.extend(random.sample(U[0:48],24))
##    T.extend(random.sample(U[48:168],60))
##    T.extend(random.sample(U[168:247],40))
##    for i in T:
##        U.remove(i)
##    return L,U,T


def wine_02_05():
    allInd = list(range(177))
    L = []
    U = []
    T = []
    L.extend(random.sample(allInd[0:58],4))
    L.extend(random.sample(allInd[58:129],5))
    L.extend(random.sample(allInd[129:177],4))
    for i in L:
        allInd.remove(i)
    U = allInd
    T.extend(random.sample(U[0:54],14))
    T.extend(random.sample(U[54:120],16))
    T.extend(random.sample(U[120:164],11))
    for i in T:
        U.remove(i)
    return L,U,T

def seeds_02_05():
    allInd = list(range(209))
    L = []
    U = []
    T = []
    L.extend(random.sample(allInd[0:69],5))
    L.extend(random.sample(allInd[69:139],5))
    L.extend(random.sample(allInd[139:209],5))
    for i in L:
        allInd.remove(i)
    U = allInd
    T.extend(random.sample(U[0:64],16))
    T.extend(random.sample(U[64:129],16))
    T.extend(random.sample(U[129:194],16))
    for i in T:
        U.remove(i)
    return L,U,T

##def get_cirdata():
##    x = np.arange(-1,1,0.01)
##    y1 = [np.sqrt(1-i**2)+random.uniform(-0.1,0.1) for i in x]
##    y2 = [-np.sqrt(1-j**2)+random.uniform(-0.1,0.1) for j in x]
####    plt.scatter(x,y1,color='r')
####    plt.scatter(x,y2,color='b')
####    plt.show()
##    label1 = np.zeros((200,1))
##    label2 = []
##    for i in label1:
##        label2.append(i+1)
##    
##    xy1 = np.dstack((x,y1))
##    xy2 = np.dstack((x,y2))
##    xy1 = xy1[0]
##    xy2 = xy2[0]
##    class1 = np.hstack((xy1,label1))
##    class2 = np.hstack((xy2,label2))
##    outdata = np.vstack((class1,class2))
##    return outdata

##def cirdata_02_05():
##    allInd = list(range(400))
##    L = []
##    U = []
##    T = []
##    L.extend(random.sample(allInd[0:200],40))
##    L.extend(random.sample(allInd[200:400],40))
##    for i in L:
##        allInd.remove(i)
##    U = allInd
##    T.extend(random.sample(U[0:160],80))
##    T.extend(random.sample(U[160:320],80))
##    for i in T:
##        U.remove(i)
##    return L,U,T

def testdata():
##    x11 = np.arange(2,6,0.05)
##    y11 = np.random.normal(2,0.5,len(x11))
##    c11 = np.dstack((x11,y11))[0]

    y12 = np.arange(1,9,0.4)
    x12 = np.random.normal(3.5,0.5,len(y12))
    c12 = np.dstack((x12,y12))[0]

##    c1 = np.vstack((c11,c12))

##    x21 = np.arange(4,8,0.05)
##    y21 = np.random.normal(8,0.5,len(x21))
##    c21 = np.dstack((x21,y21))[0]

    y22 = np.arange(1,9,0.4)
    x22 = np.random.normal(6.5,0.5,len(y22))
    c22 = np.dstack((x22,y22))[0]

##    c2 = np.vstack((c21,c22))
    return c12,c22


##c1,c2 = testdata()
##fr = open(r'C:\Users\Administrator\Desktop\UCI数据\testdataC1.xlsx','wb')
##pickle.dump(c1,fr,True)
##fr.close()
##
##fr = open(r'C:\Users\Administrator\Desktop\UCI数据\testdataC2.xlsx','wb')
##pickle.dump(c2,fr,True)
##fr.close()
##
##plt.xlim(0,10)
##plt.ylim(0,10)
##plt.scatter(c1[:,0],c1[:,1])
##plt.scatter(c2[:,0],c2[:,1])
##plt.show()

##def gettestdata():
##    fr1 = open(r'C:\Users\Administrator\Desktop\UCI数据\testdataC1.xlsx','rb')
##    c1 = pickle.load(fr1)
##    fr1.close()
##
##    fr2 = open(r'C:\Users\Administrator\Desktop\UCI数据\testdataC2.xlsx','rb')
##    c2 = pickle.load(fr2)
##    fr2.close()
##    return np.vstack((c1,c2))






