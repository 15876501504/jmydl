# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
# 读入数据，并去除索引列
data = pd.read_excel(r'C:\Users\15876\Desktop\jmydl\jmydl_data.xlsx').iloc[:, 1:]
print(u'原有数据记录长度',data.shape[0])

# 去除完全相同的数据
data.drop_duplicates(inplace=True)
print(u'去除完全相同的数据后数据长度',data.shape[0])

data.sj = pd.to_datetime(data.sj) # 将时间转换datetime格式
data = data.loc[~(data.sj.dt.year==2015),:] # 删除2015年数据
print(u'去除2015年的数据后数据长度',data.shape[0])
# 补全区局
# 删除时间或者用电量为空值的数据
data = data.loc[~(data.sj.isnull()|data.rydl_kwh.isnull()),:]
print(u'去除时间或者用电量为空值的数据',data.shape[0])

## 剔除数据"大小大"或者"小大小"情况
exceptIndex11 = data.rydl_kwh.diff(1)  # 向前差分
exceptIndex11 = exceptIndex11.fillna(1)  # 补nan值
exceptIndex12 = data.rydl_kwh.diff(-1)
exceptIndex12 = exceptIndex12.fillna(-1)
exceptIndex1 = ((exceptIndex11 > 0) & (exceptIndex12 > 0)) | ((exceptIndex11 < 0) & (exceptIndex12 < 0))
exceptIndex21 = data.yhbh.diff(1)
exceptIndex21 = exceptIndex21.fillna(0)
exceptIndex22 = data.yhbh.diff(-1)
exceptIndex22 = exceptIndex22.fillna(0)
exceptIndex2 = (exceptIndex21 == 0) & (exceptIndex22 == 0)
exceptIndex = exceptIndex1 & exceptIndex2
data = data.loc[~exceptIndex, :]

yhbh = data.yhbh.unique() # 获取用户编号
print(u'用户数目',len(yhbh))

beginDate = data.sj.min() # 开始日期
endDate = data.sj.max() # 结束日期
allDate = pd.date_range(start=beginDate,end=endDate, freq='D') # 所有日期
allDateLen = len(allDate) # 获取日期长度
# 找到时间长度缺失的用户
tmp = data.groupby('yhbh').count()['sj']
tmp = tmp[tmp!=allDateLen]
missDataUser = tmp.index
print(u'缺失数据用户数目',len(missDataUser))
# 为每个用户补缺失数据
fillMissData = pd.DataFrame()
for i in range(len(missDataUser)):
    storeDict = {}
    tmpYhbh = missDataUser[i] # 获取用户编号
    tmpIndex = data.yhbh==tmpYhbh # 获取下标
    tmpData = data.loc[tmpIndex ,:] # 所有数据
    tmpSj = tmpData.sj # 获取用户时间
    tmpRydl = tmpData.rydl_kwh # 获取用户日用电量
    tmpSsqj = tmpData.ssqj[tmpData.index[0]] # 获取所属区局
    tmpYddz = tmpData.yddz[tmpData.index[0]] # 获取用电地址
    tmpHtrlKw = tmpData.htrl_kw[tmpData.index[0]] # 获取合同容量
    missSj = allDate[~allDate.isin(tmpSj)] # 获取用户缺失时间
    while len(missSj)>0: # 判断用户缺失时间是否为空
        afterTime = tmpSj[tmpSj > missSj[0]] #
        beforeTime = tmpSj[tmpSj < missSj[0]]
        if (len(afterTime)>0)&(len(beforeTime)>0): # 前面有时间，后面有时间
            print('i:', i, u' 缺失用户:'.encode('gb2312'), tmpYhbh, u' 缺失时间:'.encode('gb2312'), missSj[0], ' flag:', 1)
            afterTime = afterTime.min() # 缺失数据的后一个时间数据
            beforeTime = beforeTime.max() # 缺失数据的前一个时间数据
            afterRydl = tmpRydl[tmpSj == afterTime] # 缺失数据的后一个时间数据的日用电量
            beforeRydl = tmpRydl[tmpSj == beforeTime] # 缺失数据的前一个时间数据的日用电
            # 填补缺失日期
            fillDate = pd.date_range(start=beforeTime + pd.Timedelta('1 days'), end=afterTime - pd.Timedelta('1 days'),freq='D')
            # 填补缺失值
            fillValue = np.linspace(start=beforeRydl, stop=afterRydl, num=len(fillDate) + 2)[1:-1]
            # 删除已经填补的日期
            missSj = missSj[~((missSj > beforeTime) & (missSj < afterTime))]
        else:
            if (len(afterTime) > 0):# 该时间的前面时间都缺失，但后面有时间数据
                print('i:',i,' 缺失用户:', tmpYhbh, ' 缺失时间:', missSj[0], ' flag:', 2)
                afterTime = afterTime.min() # 缺失数据的后一个时间数据
                afterRydl = tmpRydl[tmpSj == afterTime] # 缺失数据的后一个时间数据用电量
                # 填补缺失日期
                fillDate = pd.date_range(start=missSj[0],end=afterTime - pd.Timedelta('1 days'),freq='D')
                # 填补缺失日期的缺失值
                fillValue = np.linspace(start=afterRydl, stop=afterRydl, num=len(fillDate))
                # 删除已经填补日期
                missSj = missSj[~((missSj >=missSj[0]) & (missSj < afterTime))]
            else: # 该时间的后面时间数据都缺失，但前面没缺失
                print('i:',i,' 缺失用户:', tmpYhbh, ' 缺失时间:', missSj[0], ' flag:', 3)
                beforeTime = tmpSj[tmpSj < missSj[0]].max() # 缺失数据的前一个时间数据
                beforeRydl = tmpRydl[tmpSj == beforeTime] # 缺失数据的前一个时间的日用电量
                # 填补缺失日期
                fillDate = pd.date_range(start=beforeTime+pd.Timedelta('1 days'),end=endDate,freq='D')
                # 填补缺失值
                fillValue = np.linspace(start=beforeRydl, stop=beforeRydl, num=len(fillDate))
                # 删除已填补日期
                missSj = missSj[~((missSj >beforeTime) & (missSj <=endDate))]
        storeDict['sj'] = fillDate
        storeDict['yhbh'] = tmpYhbh
        storeDict['rydl_kwh'] = fillValue
        storeDict['ssqj'] = tmpSsqj
        storeDict['yddz'] = tmpYddz
        storeDict['htrl_kw'] = tmpHtrlKw
        fillMissData = pd.concat([fillMissData,pd.DataFrame(storeDict)])
print(u'填补数据个数:', fillMissData.shape[0])
allData = pd.concat([fillMissData,data]) # 合并数据
allData = allData.sort_values(by=['yhbh','sj']) # 根据用户编号，时间排序
allData.index = range(allData.shape[0]) # 重置索引
print(u'每个用户的时间数据长度:'.encode('gb2312'),allData.groupby(['yhbh']).count().sj.unique())

# 计算每个居民日用电量
diffRydl = allData.rydl_kwh.diff(1)
diffRydl = diffRydl.fillna(0)
judgeSameYhbh = allData.yhbh.diff(1)
judgeSameYhbh = judgeSameYhbh.fillna(0)
judgeSameYhbh = (judgeSameYhbh==0)*1
allData['diffRydl'] = diffRydl*judgeSameYhbh
allData.head()

# 经过以上操作后，还有一个用户12月1日的用电量为-0.01，直接将它变为0
allData.loc[allData.diffRydl<0,'diffRydl'] = 0

# 预处理完成后的数据写入csv
#allData.to_csv(r'C:\Users\15876\Desktop\pred_data.csv', index=False)



