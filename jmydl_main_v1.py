# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:25:28 2018

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
import matplotlib 

# 显示负号
matplotlib.rcParams['axes.unicode_minus']=False

allData = pd.read_csv(r'C:\Users\15876\Desktop\jmydl\pred_data.csv', encoding='gbk')

yhbh = allData.yhbh.unique()
TMPbh = yhbh[7] # 选择第6个用户
TMPdata = allData[allData.yhbh==TMPbh]
TMPdata.loc[:, 'sj'] = pd.to_datetime(TMPdata.sj)
#TMPdata.set_index('sj',inplace=True) # 将sj列变成索引
'''
# 平滑用电量
roll_TMPdata.diffRydl = TMPdata.diffRydl.rolling(2).mean()
plt.subplot(211)
plt.plot(roll_TMPdata.diffRydl)
plt.subplot(212)
plt.plot(TMPdata.diffRydl)
'''

# 创建新的特征
# 获得工作日和非工作日的日期
holiday = pd.read_excel(r'C:\Users\15876\Desktop\hwlyc_nationalday.xlsx')
beginDate = TMPdata.sj.min()
endDate = TMPdata.sj.max()
all_Date = pd.date_range(beginDate,endDate,freq='D')

workday = []
notworkday = []
for date in all_Date:
    # 选出周末
    if (date.weekday() == 5) or (date.weekday() == 6):
        date = str(date).split(' ')[0]
        # 选出是周末，但需要上班的日期
        if (date in list(holiday.rq)) and (list(holiday[holiday.rq==date].is_relax)==[0]):
            workday.append(date)
        else:
            notworkday.append(date)
    # 非周末
    else:
        date = str(date).split(' ')[0]
        if (date in list(holiday.rq)) and (list(holiday[holiday.rq==date].is_relax)==[1]):
            notworkday.append(date)
        else:
            workday.append(date)
    notwork_index = []
# alldate中节假日为1，工作日为0
alldate = []    
for j in range(len(all_Date)):
    date = str(all_Date[j]).split(' ')[0]
    if date in notworkday:
        alldate.append(1)# 节假日为1
    else:
        alldate.append(0)# 工作日为0
        
is_before_holiday = []
is_after_holiday = []
for i in range(1, len(alldate)):
    if alldate[i] == 1:
        is_before_holiday.append(1)
    else:
        is_before_holiday.append(0)
is_before_holiday.append(0)# 6月27日不是节假日前一天

is_after_holiday.append(0)# 1月1日不是节假日后一天
for i in range(0, len(alldate)-1):
    if alldate[i] == 1:
        is_after_holiday.append(1)
    else:
        is_after_holiday.append(0)
        
# 关于日期的特征：[一周中第几天，一年中第几天，一个月中第几天，是否月初，是否月末，
#                年，月，一年中第几周，第几季度，是否节假日，是否节假日前一天，是否节假日后一天]
TMPdata['dayofweek'] = TMPdata.sj.dt.dayofweek
TMPdata['dayofyear'] = TMPdata.sj.dt.dayofyear
TMPdata['dayofmonth'] = TMPdata.sj.dt.day
TMPdata['is_month_end'] = TMPdata.sj.dt.is_month_end
TMPdata['is_month_start'] = TMPdata.sj.dt.is_month_start
TMPdata['year'] = TMPdata.sj.dt.year
TMPdata['month'] = TMPdata.sj.dt.month
TMPdata['weekofyear'] = TMPdata.sj.dt.weekofyear
TMPdata['quarter'] = TMPdata.sj.dt.quarter
TMPdata['holiday'] = alldate
TMPdata['is_before_holiday'] = is_before_holiday
TMPdata['is_after_holiday'] = is_after_holiday

# 关于用户的特征[每个月用电量的平均数，中位数，方差（12*3=36个特征），
#              一周每一天用电量的平均数，中位数，方差（7*3=21个特征），
#              节假日和工作日用电量的平均数，中位数，方差（2*3=6个特征）]
# 平均数
TMPdata['month_avg'] = 0
TMPdata.loc[TMPdata.month==1, 'month_avg'] = np.average(TMPdata[TMPdata.month==1].diffRydl)
TMPdata.loc[TMPdata.month==2, 'month_avg'] = np.average(TMPdata[TMPdata.month==2].diffRydl)
TMPdata.loc[TMPdata.month==3, 'month_avg'] = np.average(TMPdata[TMPdata.month==3].diffRydl)
TMPdata.loc[TMPdata.month==4, 'month_avg'] = np.average(TMPdata[TMPdata.month==4].diffRydl)
TMPdata.loc[TMPdata.month==5, 'month_avg'] = np.average(TMPdata[TMPdata.month==5].diffRydl)
TMPdata.loc[TMPdata.month==6, 'month_avg'] = np.average(TMPdata[TMPdata.month==6].diffRydl)
TMPdata.loc[TMPdata.month==7, 'month_avg'] = np.average(TMPdata[TMPdata.month==7].diffRydl)
TMPdata.loc[TMPdata.month==8, 'month_avg'] = np.average(TMPdata[TMPdata.month==8].diffRydl)
TMPdata.loc[TMPdata.month==9, 'month_avg'] = np.average(TMPdata[TMPdata.month==9].diffRydl)
TMPdata.loc[TMPdata.month==10, 'month_avg'] = np.average(TMPdata[TMPdata.month==10].diffRydl)
TMPdata.loc[TMPdata.month==11, 'month_avg'] = np.average(TMPdata[TMPdata.month==11].diffRydl)
TMPdata.loc[TMPdata.month==12, 'month_avg'] = np.average(TMPdata[TMPdata.month==12].diffRydl)

TMPdata['week_avg'] = 0
TMPdata.loc[TMPdata.dayofweek==0, 'week_avg'] = np.average(TMPdata[TMPdata.dayofweek==0].diffRydl)
TMPdata.loc[TMPdata.dayofweek==1, 'week_avg'] = np.average(TMPdata[TMPdata.dayofweek==1].diffRydl)
TMPdata.loc[TMPdata.dayofweek==2, 'week_avg'] = np.average(TMPdata[TMPdata.dayofweek==2].diffRydl)
TMPdata.loc[TMPdata.dayofweek==3, 'week_avg'] = np.average(TMPdata[TMPdata.dayofweek==3].diffRydl)
TMPdata.loc[TMPdata.dayofweek==4, 'week_avg'] = np.average(TMPdata[TMPdata.dayofweek==4].diffRydl)
TMPdata.loc[TMPdata.dayofweek==5, 'week_avg'] = np.average(TMPdata[TMPdata.dayofweek==5].diffRydl)
TMPdata.loc[TMPdata.dayofweek==6, 'week_avg'] = np.average(TMPdata[TMPdata.dayofweek==6].diffRydl)

TMPdata['is_holiday_avg'] = 0
TMPdata.loc[TMPdata.holiday==1, 'is_holiday_avg'] = np.average(TMPdata[TMPdata.holiday==1].diffRydl)
TMPdata.loc[TMPdata.holiday==0, 'is_holiday_avg'] = np.average(TMPdata[TMPdata.holiday==0].diffRydl)
# 中位数
TMPdata['month_median'] = 0
TMPdata.loc[TMPdata.month==1, 'month_median'] = np.median(TMPdata[TMPdata.month==1].diffRydl)
TMPdata.loc[TMPdata.month==2, 'month_median'] = np.median(TMPdata[TMPdata.month==2].diffRydl)
TMPdata.loc[TMPdata.month==3, 'month_median'] = np.median(TMPdata[TMPdata.month==3].diffRydl)
TMPdata.loc[TMPdata.month==4, 'month_median'] = np.median(TMPdata[TMPdata.month==4].diffRydl)
TMPdata.loc[TMPdata.month==5, 'month_median'] = np.median(TMPdata[TMPdata.month==5].diffRydl)
TMPdata.loc[TMPdata.month==6, 'month_median'] = np.median(TMPdata[TMPdata.month==6].diffRydl)
TMPdata.loc[TMPdata.month==7, 'month_median'] = np.median(TMPdata[TMPdata.month==7].diffRydl)
TMPdata.loc[TMPdata.month==8, 'month_median'] = np.median(TMPdata[TMPdata.month==8].diffRydl)
TMPdata.loc[TMPdata.month==9, 'month_median'] = np.median(TMPdata[TMPdata.month==9].diffRydl)
TMPdata.loc[TMPdata.month==10, 'month_median'] = np.median(TMPdata[TMPdata.month==10].diffRydl)
TMPdata.loc[TMPdata.month==11, 'month_median'] = np.median(TMPdata[TMPdata.month==11].diffRydl)
TMPdata.loc[TMPdata.month==12, 'month_median'] = np.median(TMPdata[TMPdata.month==12].diffRydl)

TMPdata['week_median'] = 0
TMPdata.loc[TMPdata.dayofweek==0, 'week_median'] = np.median(TMPdata[TMPdata.dayofweek==0].diffRydl)
TMPdata.loc[TMPdata.dayofweek==1, 'week_median'] = np.median(TMPdata[TMPdata.dayofweek==1].diffRydl)
TMPdata.loc[TMPdata.dayofweek==2, 'week_median'] = np.median(TMPdata[TMPdata.dayofweek==2].diffRydl)
TMPdata.loc[TMPdata.dayofweek==3, 'week_median'] = np.median(TMPdata[TMPdata.dayofweek==3].diffRydl)
TMPdata.loc[TMPdata.dayofweek==4, 'week_median'] = np.median(TMPdata[TMPdata.dayofweek==4].diffRydl)
TMPdata.loc[TMPdata.dayofweek==5, 'week_median'] = np.median(TMPdata[TMPdata.dayofweek==5].diffRydl)
TMPdata.loc[TMPdata.dayofweek==6, 'week_median'] = np.median(TMPdata[TMPdata.dayofweek==6].diffRydl)

TMPdata['is_holiday_median'] = 0
TMPdata.loc[TMPdata.holiday==1, 'is_holiday_median'] = np.median(TMPdata[TMPdata.holiday==1].diffRydl)
TMPdata.loc[TMPdata.holiday==0, 'is_holiday_median'] = np.median(TMPdata[TMPdata.holiday==0].diffRydl)

# 方差
TMPdata['month_std'] = 0
TMPdata.loc[TMPdata.month==1, 'month_std'] = np.std(TMPdata[TMPdata.month==1].diffRydl)
TMPdata.loc[TMPdata.month==2, 'month_std'] = np.std(TMPdata[TMPdata.month==2].diffRydl)
TMPdata.loc[TMPdata.month==3, 'month_std'] = np.std(TMPdata[TMPdata.month==3].diffRydl)
TMPdata.loc[TMPdata.month==4, 'month_std'] = np.std(TMPdata[TMPdata.month==4].diffRydl)
TMPdata.loc[TMPdata.month==5, 'month_std'] = np.std(TMPdata[TMPdata.month==5].diffRydl)
TMPdata.loc[TMPdata.month==6, 'month_std'] = np.std(TMPdata[TMPdata.month==6].diffRydl)
TMPdata.loc[TMPdata.month==7, 'month_std'] = np.std(TMPdata[TMPdata.month==7].diffRydl)
TMPdata.loc[TMPdata.month==8, 'month_std'] = np.std(TMPdata[TMPdata.month==8].diffRydl)
TMPdata.loc[TMPdata.month==9, 'month_std'] = np.std(TMPdata[TMPdata.month==9].diffRydl)
TMPdata.loc[TMPdata.month==10, 'month_std'] = np.std(TMPdata[TMPdata.month==10].diffRydl)
TMPdata.loc[TMPdata.month==11, 'month_std'] = np.std(TMPdata[TMPdata.month==11].diffRydl)
TMPdata.loc[TMPdata.month==12, 'month_std'] = np.std(TMPdata[TMPdata.month==12].diffRydl)

TMPdata['week_std'] = 0
TMPdata.loc[TMPdata.dayofweek==0, 'week_std'] = np.std(TMPdata[TMPdata.dayofweek==0].diffRydl)
TMPdata.loc[TMPdata.dayofweek==1, 'week_std'] = np.std(TMPdata[TMPdata.dayofweek==1].diffRydl)
TMPdata.loc[TMPdata.dayofweek==2, 'week_std'] = np.std(TMPdata[TMPdata.dayofweek==2].diffRydl)
TMPdata.loc[TMPdata.dayofweek==3, 'week_std'] = np.std(TMPdata[TMPdata.dayofweek==3].diffRydl)
TMPdata.loc[TMPdata.dayofweek==4, 'week_std'] = np.std(TMPdata[TMPdata.dayofweek==4].diffRydl)
TMPdata.loc[TMPdata.dayofweek==5, 'week_std'] = np.std(TMPdata[TMPdata.dayofweek==5].diffRydl)
TMPdata.loc[TMPdata.dayofweek==6, 'week_std'] = np.std(TMPdata[TMPdata.dayofweek==6].diffRydl)

TMPdata['is_holiday_std'] = 0
TMPdata.loc[TMPdata.holiday==1, 'is_holiday_std'] = np.median(TMPdata[TMPdata.holiday==1].diffRydl)
TMPdata.loc[TMPdata.holiday==0, 'is_holiday_std'] = np.median(TMPdata[TMPdata.holiday==0].diffRydl)

# 特征选择
# 删除合同容量，日用电量，所属区局，用电地址，用户编号
TMPdata.drop(['htrl_kw', 'rydl_kwh', 'ssqj', 'yddz', 'yhbh'], axis=1, inplace=True)
print("现在总共有%d个特征"%len(TMPdata.columns))

# 标准化，将数据特征全部化为0-1
min_max_scaler = preprocessing.MinMaxScaler()
TMPdata.iloc[:, 2:] = min_max_scaler.fit_transform(TMPdata.iloc[:, 2:])
# PCA降维
from sklearn.decomposition import PCA
pca = PCA(n_components=0.90) # 保留90%的信息
pca.fit(TMPdata.iloc[:, 2:])
#print(pca.explained_variance_ratio_)
#print(pca.explained_variance_)

new_TMPdata = pca.transform(TMPdata.iloc[:, 2:])# pca降维后数据
print("PCA降维后，特征数目：%d"%new_TMPdata.shape[1])
# 划分训练集和测试集
# 将2017年6月以前的数据作为训练集，6月份的数据作为测试集
train_X = new_TMPdata[:-27, :]
train_Y = np.array(TMPdata.iloc[:-27, 1])
test_X = new_TMPdata[-27:, :]
test_Y = np.array(TMPdata.iloc[-27:, 1])


# 决策树回归
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(max_depth=8)
clf = clf.fit(train_X, train_Y)
predict_train = clf.predict(train_X) # 对训练样本进行预测
predict_test = clf.predict(test_X) # 对测试样本进行测试
print(clf.score(train_X, train_Y))
print(clf.score(test_X, test_Y))
plt.subplot(211)
plt.plot(train_Y, label='train_y')
plt.plot(predict_train, label='predict_train')
plt.subplot(212)
plt.plot(test_Y, label='test_y')
plt.plot(predict_test, label='predict_test')






