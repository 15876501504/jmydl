# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:23:29 2018

@author: 15876
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib 
# 显示负号
matplotlib.rcParams['axes.unicode_minus']=False


data = pd.read_csv(r'C:\Users\15876\Desktop\jmydl\pred_data.csv', encoding='gbk')


# 画不同合同容量的数量占比饼图
htrl_4 = len(data[data.htrl_kw==4].yhbh.unique())
htrl_6 = len(data[data.htrl_kw==6].yhbh.unique())
htrl_30 = len(data[data.htrl_kw==30].yhbh.unique())
htrl_150 = len(data[data.htrl_kw==150].yhbh.unique())
htrl_20 = len(data[data.htrl_kw==20].yhbh.unique())
htrl_12 = len(data[data.htrl_kw==12].yhbh.unique())
htrl_5 = len(data[data.htrl_kw==5].yhbh.unique())
htrl_2 = len(data[data.htrl_kw==2].yhbh.unique())

labels = [4,6,5,30,150,20,12,2]
patches,l_text,p_text = plt.pie([htrl_4,htrl_6,htrl_5,htrl_30,htrl_150,htrl_20,htrl_12,htrl_2], labels=labels, autopct='%1.2f%%')
plt.title('不同合同容量用户数占比', fontsize=20)
for t in l_text:
    t.set_size=(30)
for t in p_text:
    t.set_size=(20)
# 设置x，y轴刻度一致，这样饼图才能是圆的
plt.axis('equal')
plt.legend()


# 画不同合同容量用户总用电量占比饼图
print(data.htrl_kw.unique())

htrl_4_rydl = sum(data[data.htrl_kw==4].diffRydl)
htrl_6_rydl = sum(data[data.htrl_kw==6].diffRydl)
htrl_30_rydl = sum(data[data.htrl_kw==30].diffRydl)
htrl_150_rydl = sum(data[data.htrl_kw==150].diffRydl)
htrl_20_rydl = sum(data[data.htrl_kw==20].diffRydl)
htrl_12_rydl = sum(data[data.htrl_kw==12].diffRydl)
htrl_5_rydl = sum(data[data.htrl_kw==5].diffRydl)
htrl_2_rydl = sum(data[data.htrl_kw==2].diffRydl)

labels = [4,6,5,30,150,20,12,2]
plt.pie([htrl_4_rydl,htrl_6_rydl,htrl_5_rydl,htrl_30_rydl,htrl_150_rydl,htrl_20_rydl,htrl_12_rydl,htrl_2_rydl], labels=labels, autopct='%1.2f%%')
plt.title('不同合同容量用户总用电量占比', fontsize=20)
for t in l_text:
    t.set_size=(30)
for t in p_text:
    t.set_size=(20)
# 设置x，y轴刻度一致，这样饼图才能是圆的
plt.axis('equal')
plt.legend()


# 区局
print(data.ssqj.unique())
th_num = len(data[data.ssqj=='天河'].yhbh.unique())
hd_num = len(data[data.ssqj=='花都'].yhbh.unique())
py_num = len(data[data.ssqj=='番禺'].yhbh.unique())
nan_num = len(data[data.ssqj.isnull()].yhbh.unique())

labels = ['天河', '花都', '番禺', 'nan']
plt.pie([th_num, hd_num, py_num, nan_num], labels=labels, autopct='%1.2f%%')
plt.title('不同区局用户数占比', fontsize=20)
for t in l_text:
    t.set_size=(30)
for t in p_text:
    t.set_size=(20)
# 设置x，y轴刻度一致，这样饼图才能是圆的
plt.axis('equal')
plt.legend()


th_rydl = sum(data[data.ssqj=='天河'].diffRydl)
hd_rydl = sum(data[data.ssqj=='花都'].diffRydl)
py_rydl = sum(data[data.ssqj=='番禺'].diffRydl)
nan_rydl = sum(data[data.ssqj.isnull()].diffRydl)

labels = ['天河', '花都', '番禺', 'nan']
plt.pie([th_rydl, hd_rydl, py_rydl, nan_rydl], labels=labels, autopct='%1.2f%%')
plt.title('不同区局用户总用电量占比', fontsize=20)
for t in l_text:
    t.set_size=(30)
for t in p_text:
    t.set_size=(20)
# 设置x，y轴刻度一致，这样饼图才能是圆的
plt.axis('equal')
plt.legend()


# 暂时不考虑用电地址
# data['filter_yddz'] = data.apply(lambda x: ''.join(list(filter(lambda y: y not in '0123456789', x[4]))), axis=1)


# 日期
data.sj = pd.to_datetime(data.sj)# 将sj转化为日期格式
la = list(data.sj.dt.date)# x轴刻度只显示年月日
yhbh = data.yhbh.unique()# 得到用户编号
# 画前6个用户的用电量变化情况
for i in range(6):
    tmpbh = yhbh[i]
    tmpdata = data[data.yhbh==tmpbh]
    plt.subplot(3,2,i+1)
    plt.plot(range(544), tmpdata.diffRydl)
    plt.xticks([0,100,200,300,400,500])
    plt.xticks([0, 100, 200, 300, 400, 500], [la[0], la[100], la[200], la[300], la[400], la[500]])
    plt.title("图2.%d, 用户%s的用电情况"%(i+1,tmpbh))
plt.tight_layout()


data['year'] = data.sj.dt.year
data['month'] = data.sj.dt.month
data['day'] = data.sj.dt.day


# 画总用电量随月份的变化。
month_rydl = []
beginDate = data.sj.min()
endDate = data.sj.max()
all_month = pd.date_range(beginDate, endDate, freq='M').month

# append2016年12个月用电量
for i in range(1, 13):
    month_rydl.append(sum(data[(data.year==2016)&(data.month==i)].diffRydl))
# append2017年1-6月用电量
for i in range(1, 7):
    month_rydl.append(sum(data[(data.year==2017)&(data.month==i)].diffRydl))

plt.plot(range(1, 19), month_rydl)
plt.title("所有用户月用电总量", fontsize=20)
plt.xticks(range(1, 19), all_month, fontsize=20)


# 画第六个到低12个用户月用电量的变化（索引从1开始）
for i in range(6, 12):
    tmpbh = yhbh[i]
    tmpdata = data[data.yhbh==tmpbh]
    month_data = tmpdata[['year', 'month', 'diffRydl']].groupby(['year', 'month'],as_index=False).agg(['sum']).reset_index()
    plt.subplot(3,2,i-5)
    plt.plot(range(1, 19), month_data['diffRydl'])
    plt.title("图3.%d,\n用户%d月用电总量"%(i-5, tmpbh), fontsize=20)
    plt.xticks(range(1, 19), all_month, fontsize=20)
plt.tight_layout()



# 工作日与非工作日
holiday = pd.read_excel(r'C:\Users\15876\Desktop\hwlyc_nationalday.xlsx')
beginDate = data.sj.min()
endDate = data.sj.max()
all_Date = pd.date_range(beginDate,endDate,freq='D')

# 获得工作日和非工作日的日期
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

for i in range(6, 12):
    notwork_index = []
    for j in range(len(all_Date)):
        date = str(all_Date[j]).split(' ')[0]
        if date in notworkday:
            notwork_index.append(j)
    
    tmpbh = yhbh[i]
    tmpdata = data[data.yhbh==tmpbh]
    w_data = []
    n_data = []
    for m in range(len(allDate)):
        if str(tmpdata.iloc[m, 2]).split(' ')[0] in workday:
            w_data.append(tmpdata.iloc[m, 6])
        else:
            n_data.append(tmpdata.iloc[m, 6])
            
    plt.subplot(3,2,i-5)
    plt.plot(range(544), tmpdata.diffRydl, label='日用电量')
    print("用户编号%d，工作日日均用电量%.2f, 非工作日日均用电量%.2f" %(tmpbh, sum(w_data)/len(w_data), sum(n_data)/len(n_data)))
    plt.xticks([0,100,200,300,400,500])
    plt.xticks([0, 100, 200, 300, 400, 500], [la[0], la[100], la[200], la[300], la[400], la[500]])
    plt.title("用户%s的用电情况"%tmpbh)
    plt.scatter(notwork_index, n_data, color='r', s=10, label='非工作日用电量')
    plt.legend(fontsize=14)
    
plt.tight_layout()
















