# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:25:11 2018

@author: 15876
"""

import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = pd.read_csv(r'C:\Users\15876\Desktop\jmydl\pred_data.csv', encoding='gbk')
data.sj = pd.to_datetime(data.sj)
data.sort_values(['yhbh', 'sj'], inplace=True)
all_yhbh = data.yhbh.unique()

cluster_data = []
c = 1
for TMPbh in all_yhbh:
    TMPdata = data[data.yhbh == TMPbh]
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
    
    TMPdata = TMPdata.sort_values('sj')  

    # 温度，天气特征
    add_data = pd.read_csv(r'C:\Users\15876\Desktop\weather_finish_dealing(1).csv')
    add_data.date = pd.to_datetime(add_data.date)
    add_data_max = add_data[['date', 'max_wd', 'max_csd', 'max_fs']].groupby(['date'], as_index=False).max()
    add_data_min = add_data[['date', 'min_wd', 'min_csd', 'min_fs']].groupby(['date'], as_index=False).min()
    add_data_avg = add_data[['date', 'avg_wd', 'avg_csd', 'avg_fs']].groupby(['date'], as_index=False).mean()
    new_add_data = pd.merge(add_data_max, add_data_min, on='date')
    new_add_data = pd.merge(new_add_data, add_data_avg, on='date')
    new_add_data = new_add_data.sort_values('date')
    # 写入csv
    #new_add_data.to_csv(r'C:\Users\15876\Desktop\jmydl\pred_weather', index=False)
    TMPdata = pd.merge(TMPdata, new_add_data, left_on='sj', right_on='date', how='left')
    

    # 关于日期的特征：[一周中第几天，一年中第几天，一个月中第几天，是否月初，是否月末，
    #                年，月，一年中第几周，第几季度，是否节假日，是否节假日前一天，
    #                是否节假日后一天, ]
    TMPdata['dayofweek'] = TMPdata.sj.dt.dayofweek
    TMPdata['dayofyear'] = TMPdata.sj.dt.dayofyear
    TMPdata['dayofmonth'] = TMPdata.sj.dt.day
    #TMPdata['is_month_end'] = TMPdata.sj.dt.is_month_end
    #TMPdata['is_month_start'] = TMPdata.sj.dt.is_month_start
    TMPdata['year'] = TMPdata.sj.dt.year
    TMPdata['month'] = TMPdata.sj.dt.month
    #TMPdata['weekofyear'] = TMPdata.sj.dt.weekofyear
    #TMPdata['quarter'] = TMPdata.sj.dt.quarter
    TMPdata['holiday'] = alldate
    TMPdata['is_before_holiday'] = is_before_holiday
    TMPdata['is_after_holiday'] = is_after_holiday    
    
    
    
        
    year_mean = np.mean(TMPdata.diffRydl)
    year_std = np.std(TMPdata.diffRydl)
    month1_mean = np.mean(TMPdata[TMPdata.month==1].diffRydl)
    month1_std = np.std(TMPdata[TMPdata.month==1].diffRydl)
    month2_mean = np.mean(TMPdata[TMPdata.month==2].diffRydl)
    month2_std = np.std(TMPdata[TMPdata.month==2].diffRydl)
    month3_mean = np.mean(TMPdata[TMPdata.month==3].diffRydl)
    month3_std = np.std(TMPdata[TMPdata.month==3].diffRydl)
    month4_mean = np.mean(TMPdata[TMPdata.month==4].diffRydl)
    month4_std = np.std(TMPdata[TMPdata.month==4].diffRydl)
    month5_mean = np.mean(TMPdata[TMPdata.month==5].diffRydl)
    month5_std = np.std(TMPdata[TMPdata.month==5].diffRydl)
    month6_mean = np.mean(TMPdata[TMPdata.month==6].diffRydl)
    month6_std = np.std(TMPdata[TMPdata.month==6].diffRydl)
    month7_mean = np.mean(TMPdata[TMPdata.month==7].diffRydl)
    month7_std = np.std(TMPdata[TMPdata.month==7].diffRydl)
    month8_mean = np.mean(TMPdata[TMPdata.month==8].diffRydl)
    month8_std = np.std(TMPdata[TMPdata.month==8].diffRydl)
    month9_mean = np.mean(TMPdata[TMPdata.month==9].diffRydl)
    month9_std = np.std(TMPdata[TMPdata.month==9].diffRydl)
    month10_mean = np.mean(TMPdata[TMPdata.month==10].diffRydl)
    month10_std = np.std(TMPdata[TMPdata.month==10].diffRydl)
    month11_mean = np.mean(TMPdata[TMPdata.month==11].diffRydl)
    month11_std = np.std(TMPdata[TMPdata.month==11].diffRydl)
    month12_mean = np.mean(TMPdata[TMPdata.month==12].diffRydl)
    month12_std = np.std(TMPdata[TMPdata.month==12].diffRydl)
    holiday_mean = np.mean(TMPdata[TMPdata.holiday==1].diffRydl)
    holiday_std = np.std(TMPdata[TMPdata.holiday==1].diffRydl)
    not_holiday_mean = np.mean(TMPdata[TMPdata.holiday==0].diffRydl)
    not_holiday_std = np.std(TMPdata[TMPdata.holiday==0].diffRydl)
    high_wd_mean = np.mean(TMPdata[TMPdata.avg_wd>32].diffRydl)
    high_wd_std = np.std(TMPdata[TMPdata.avg_wd>32].diffRydl)
    low_wd_mean = np.mean(TMPdata[TMPdata.avg_wd<10].diffRydl)
    low_wd_std = np.std(TMPdata[TMPdata.avg_wd<10].diffRydl)
    cluster_data.append([year_mean, year_std, month1_mean, month1_std, month2_mean,\
                         month2_std, month3_mean, month3_std, month4_mean, month4_std,\
                         month5_mean, month5_std, month6_mean, month6_std, month7_mean,\
                         month7_std, month8_mean, month8_std, month9_mean, month9_std,\
                         month10_mean, month10_std, month11_mean, month11_std, month12_mean,\
                         month12_std, holiday_mean, holiday_std, not_holiday_mean, not_holiday_std,\
                         high_wd_mean, high_wd_std, low_wd_mean, low_wd_std])
    print(c)
    c += 1

cludata = np.array(cluster_data)
print("数据维度：%d"%(cludata.shape[1]))
# PCA降维
from sklearn.decomposition import PCA
new_clu_data = PCA(n_components=0.9).fit_transform(cludata)
print("PCA降维后数据维度：%d"%(new_clu_data.shape[1]))

# 肘部法则找出最佳聚类数
'''    
inertia = []
for i in range(1, 21):
    clf = KMeans(n_clusters=i)
    clf.fit(new_clu_data)
    inertia.append(clf.inertia_)
plt.plot(range(1, 21), inertia)
'''
clf = KMeans(n_clusters=5)
clf.fit(new_clu_data)
result = clf.labels_.tolist()

clu1 = []
clu2 = []
clu3 = []
clu4 = []
clu5 = []
for i in range(len(result)):
    if result[i] == 0:
        clu1.append(all_yhbh[i])
    elif result[i] == 1:
        clu2.append(all_yhbh[i])
    elif result[i] == 2:
        clu3.append(all_yhbh[i])
    elif result[i] == 3:
        clu4.append(all_yhbh[i])
    elif result[i] == 4:
        clu5.append(all_yhbh[i])

# 保存分类信息
'''
import pickle
fr = open(r'C:\Users\15876\Desktop\cluster_data.txt', 'wb')
pickle.dump([clu1, clu2, clu3, clu4, clu5], fr, True)
fr.close()
'''













