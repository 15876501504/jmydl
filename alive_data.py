# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:08:09 2018

@author: 15876
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import  stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from sklearn import preprocessing
from sklearn import metrics
import matplotlib 
import time
from warnings import filterwarnings
filterwarnings('ignore')


allData = pd.read_csv(r'C:\Users\15876\Desktop\jmydl\pred_data.csv', encoding='gbk')
allData.loc[:, 'sj'] = pd.to_datetime(allData.sj)
print(len(allData.yhbh.unique()))

# 删除最后一个月日用电量为0或日用电量小于2的用户
yhbh = allData.yhbh.unique()
drop_index = []
for TMPbh in yhbh:
    TMPdata = allData[allData.yhbh==TMPbh]
    if (np.mean(TMPdata.iloc[-30:, -1])<2):
        print(TMPbh)
        drop_index.extend(list(allData[allData.yhbh==TMPbh].index))
        #print(len(allData))
allData.drop(drop_index, axis=0, inplace=True)
print(len(allData.yhbh.unique()))
allData.reset_index(drop=True)

# 删除最后30天方差小于1的用户
yhbh = allData.yhbh.unique()
drop_index = []
for TMPbh in yhbh:
    TMPdata = allData[allData.yhbh==TMPbh]
    if (np.std(TMPdata.iloc[-30:, -1])<1):
        print(TMPbh)
        drop_index.extend(list(allData[allData.yhbh==TMPbh].index))
        #print(len(allData))
allData.drop(drop_index, axis=0, inplace=True)
print(len(allData.yhbh.unique()))
allData.reset_index(drop=True)

allData.sort_values(['yhbh', 'sj'])
allData.to_csv(r'C:\Users\15876\Desktop\jmydl\alive_data.csv', index=False)





