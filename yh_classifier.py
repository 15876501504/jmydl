# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:10:07 2018

@author: 15876
"""

import pandas as pd
import numpy as np
import time

data = pd.read_csv(r'C:\Users\15876\Desktop\jmydl\pred_data.csv', encoding='gbk')
data.sj = pd.to_datetime(data.sj)
data.sort_values(['yhbh', 'sj'], inplace=True)
all_yhbh = data.yhbh.unique()
print('用户数：%d'%len(all_yhbh))

# 选出最后30天中，平均日用电量不足2的用户
last30_mean_lower_2 = []
for TMPbh in all_yhbh:
    TMPdata = data[data.yhbh==TMPbh]
    if (np.mean(TMPdata.iloc[-30:, -1])<2):
        last30_mean_lower_2.append(TMPbh)

print(len(last30_mean_lower_2))


# 选出最后30天中，日用电量方差不足1的用户
last30_std_lower_2 = []
for TMPbh in all_yhbh:
    TMPdata = data[data.yhbh==TMPbh]
    if (np.std(TMPdata.iloc[-30:, -1])<1):
        last30_std_lower_2.append(TMPbh)

print(len(last30_std_lower_2))









