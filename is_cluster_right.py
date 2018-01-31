# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:01:46 2018

@author: 15876
"""

import jmydl_cluster
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
# 取消warning
from warnings import filterwarnings
filterwarnings('ignore')

# 显示负号
matplotlib.rcParams['axes.unicode_minus']=False

# 输入df和划窗天数， 返回划窗后的mean，std，min，max，median
def look_back(df, days):
    post_mean = [TMPdata.iloc[0, 6]]
    post_std = [TMPdata.iloc[0, 6]]
    post_min = [TMPdata.iloc[0, 6]]
    post_max = [TMPdata.iloc[0, 6]]
    post_median = [TMPdata.iloc[0, 6]]
    for i in range(len(df)):
        if (i<=(days-1)) and (i>=1):
            post_mean.append(np.average(TMPdata.iloc[:i, 6]))
            post_std.append(np.std(TMPdata.iloc[:i, 6]))
            post_max.append(np.max(TMPdata.iloc[:i, 6]))
            post_min.append(np.min(TMPdata.iloc[:i, 6]))
            post_median.append(np.median(TMPdata.iloc[:i, 6]))
        elif i>(days-1):
            post_mean.append(np.average(TMPdata.iloc[i-days:i, 6]))
            post_std.append(np.std(TMPdata.iloc[i-days:i, 6]))
            post_max.append(np.max(TMPdata.iloc[i-days:i, 6]))
            post_min.append(np.min(TMPdata.iloc[i-days:i, 6]))
            post_median.append(np.median(TMPdata.iloc[i-days:i, 6]))
    return post_mean, post_std, post_min, post_max, post_median


def get_feture(TMPdata):
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
    
    
    # 划窗产生更多特征  
    # 前一天的用电量      
    post1_rydl = []
    post1_rydl.append(TMPdata.iloc[0, 6])# 第一天的前一天用电量为自己
    for i in range(len(TMPdata)-1):
        post1_rydl.append(TMPdata.iloc[i, 6])
    
    # 前3天用电量的均值，方差，中位数，最大值，最小值
    post3_mean = [TMPdata.iloc[0, 6]]
    post3_std = [TMPdata.iloc[0, 6]]
    post3_min = [TMPdata.iloc[0, 6]]
    post3_max = [TMPdata.iloc[0, 6]]    
    for i in range(len(TMPdata)):
        if (i<=2) and (i>=1):
            post3_mean.append(np.average(TMPdata.iloc[:i, 6]))
            post3_std.append(np.std(TMPdata.iloc[:i, 6]))
            post3_max.append(np.max(TMPdata.iloc[:i, 6]))
            post3_min.append(np.min(TMPdata.iloc[:i, 6]))
        elif i>2:
            post3_mean.append(np.average(TMPdata.iloc[i-3:i, 6]))
            post3_std.append(np.std(TMPdata.iloc[i-3:i, 6]))
            post3_max.append(np.max(TMPdata.iloc[i-3:i, 6]))
            post3_min.append(np.min(TMPdata.iloc[i-3:i, 6]))
    
    
    # 7天划窗            
    post7_mean, post7_std, post7_min, post7_max, post7_median = look_back(TMPdata, 7)
    # 30天划窗
    post30_mean, post30_std, post30_min, post30_max, post30_median = look_back(TMPdata, 30)
    # 60天划窗
    post60_mean, post60_std, post60_min, post60_max, post60_median = look_back(TMPdata, 60)
    # 180天划窗
    #post180_mean, post180_std, post180_min, post180_max, post180_median = look_back(TMPdata, 180)
    
    
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
    #TMPdata['post1_rydl'] = post1_rydl
    TMPdata['post1_rydl'] = post1_rydl
    TMPdata['post3_mean'] = post3_mean
    TMPdata['post3_std'] = post3_std
    TMPdata['post3_min'] = post3_min
    TMPdata['post3_max'] = post3_max
    TMPdata['post7_mean'] = post7_mean
    TMPdata['post7_std'] = post7_std
    TMPdata['post7_min'] = post7_min
    TMPdata['post7_max'] = post7_max
    TMPdata['post7_median'] = post7_median
    TMPdata['post30_mean'] = post30_mean
    TMPdata['post30_std'] = post30_std
    TMPdata['post30_min'] = post30_min
    TMPdata['post30_max'] = post30_max
    TMPdata['post30_median'] = post30_median
    TMPdata['post60_mean'] = post60_mean
    TMPdata['post60_std'] = post60_std
    TMPdata['post60_min'] = post60_min
    TMPdata['post60_max'] = post60_max
    TMPdata['post60_median'] = post60_median
    #TMPdata['post180_mean'] = post180_mean
    #TMPdata['post180_std'] = post180_std
    #TMPdata['post180_min'] = post180_min
    #TMPdata['post180_max'] = post180_max
    #TMPdata['post180_median'] = post180_median
    
    # 特征选择
    # 删除合同容量，日用电量，所属区局，用电地址，用户编号
    TMPdata.drop(['htrl_kw', 'rydl_kwh', 'ssqj', 'yddz', 'yhbh', 'date'], axis=1, inplace=True)
    print("现在总共有%d个特征"%len(TMPdata.columns))
    X_ = TMPdata.iloc[:, 2:]
    Y_ = TMPdata.iloc[:, 1]
    '''
    # 数据转化到0-1
    from sklearn.preprocessing import MinMaxScaler
    X_ = MinMaxScaler().fit_transform(X_)
    '''
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_ = scaler.fit_transform(X_)
    
    '''
    # 使用方差选择特征
    from sklearn.feature_selection import VarianceThreshold
    new_feture_VAR = VarianceThreshold(threshold=0.1).fit_transform(X_) # 方差大于0.1的特征
    print("利用方差选择后剩下%d个特征"%new_feture_VAR.shape[1])
    '''
    '''
    # 皮尔逊相关系数
    from sklearn.feature_selection import SelectKBest
    from scipy.stats import pearsonr
    for i in range(24):
        print(pearsonr(X_[:, i], Y_))
    '''
    # 基于树模型的特征选择
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import GradientBoostingRegressor
     
    #GBDT作为基模型的特征选择
    new_feture_GBDT = SelectFromModel(GradientBoostingRegressor(max_depth=3)).fit_transform(X_, Y_)
    print("利用GBDT选择后剩下%d个特征"%new_feture_GBDT.shape[1])
    '''
    # PCA降维
    from sklearn.decomposition import PCA
    new_feture_PCA = PCA(n_components=0.9).fit_transform(X_)
    '''
    # 划分训练集和测试集
    # 将2017年6月以前的数据作为训练集，6月份的数据作为测试集
    
    X_train = new_feture_GBDT[:-27]
    y_train = np.array(Y_.iloc[:-27])
    X_test = new_feture_GBDT[-27:]
    y_test = np.array(Y_.iloc[-27:])
    
    return X_train, y_train, X_test, y_test
  
from sklearn.neural_network import MLPRegressor
    
allData = pd.read_csv(r'C:\Users\15876\Desktop\jmydl\pred_data.csv', encoding='gbk')
all_time = time.time()
yhbh = allData.yhbh.unique()
'''
import pickle
#clu1, clu2, clu3, clu4, clu5 = jmydl_cluster.get_cluster_sets()
fr = open(r'C:\Users\15876\Desktop\cluster_data.txt','rb')
[clu1, clu2, clu3, clu4, clu5] = pickle.load(fr)
fr.close()

from sklearn import metrics
def clu_r2_score(clu, c=1):
    s_time = time.time()
    r2_score = 0
    for TMPbh in clu:
        TMPdata = allData[allData.yhbh==TMPbh]
        TMPdata.loc[:, 'sj'] = pd.to_datetime(TMPdata.sj)
        #TMPdata.set_index('sj',inplace=True) # 将sj列变成索引
        X_train, y_train, X_test, y_test = get_feture(TMPdata)
        MLPR = MLPRegressor()
        MLPR.fit(X_train, y_train)
        y_test = y_test[:-1]
        predict = MLPR.predict(X_test)[1:]
        score = metrics.r2_score(y_test, predict)
        r2_score += score
        print(c)
        c += 1
    print(time.time() - s_time)
    return r2_score/len(clu)

clu1_score = clu_r2_score(clu1)
clu2_score = clu_r2_score(clu2)
clu3_score = clu_r2_score(clu3)
clu4_score = clu_r2_score(clu4)
clu5_score = clu_r2_score(clu5)

print("clu1的平均score:%.4f"%(clu1_score))
print("clu2的平均score:%.4f"%(clu2_score))
print("clu3的平均score:%.4f"%(clu3_score))
print("clu4的平均score:%.4f"%(clu4_score))
print("clu5的平均score:%.4f"%(clu5_score))
'''

mlp_r2_list = []
for TMPbh in yhbh:
    s_time = time.time()
    TMPdata = allData[allData.yhbh==TMPbh]
    TMPdata.loc[:, 'sj'] = pd.to_datetime(TMPdata.sj)
    #TMPdata.set_index('sj',inplace=True) # 将sj列变成索引
    X_train, y_train, X_test, y_test = get_feture(TMPdata)
    MLPR = MLPRegressor()
    MLPR.fit(X_train, y_train)
    y_test = y_test[:-1]
    predict = MLPR.predict(X_test)[1:]
    #mse = metrics.mean_squared_error((y_test), (predict))
    #mean = -(np.mean((y_test)))
    r2 = metrics.r2_score(y_test, predict)
    mlp_r2_list.append(r2)
    print(time.time() - s_time)
    
    
    
iso_r2_list = []
for TMPbh in yhbh:
    s_time = time.time()
    TMPdata = allData[allData.yhbh==TMPbh]
    TMPdata.loc[:, 'sj'] = pd.to_datetime(TMPdata.sj)
    #TMPdata.set_index('sj',inplace=True) # 将sj列变成索引
    X_train, y_train, X_test, y_test = get_feture(TMPdata)
    iso = IsotonicRegression()
    iso.fit(X_train, y_train)
    y_test = y_test[:-1]
    predict = iso.predict(X_test)[1:]
    #mse = metrics.mean_squared_error((y_test), (predict))
    #mean = -(np.mean((y_test)))
    r2 = metrics.r2_score(y_test, predict)
    iso_r2_list.append(r2)
    print(time.time() - s_time)    
    
    
    
    
    
    
    
    
    
    
    
    
    

