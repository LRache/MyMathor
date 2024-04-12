import pandas as pd

import pmdarima as pm


orignalData = pd.read_csv("./data1.csv", encoding="gb2312", parse_dates=["日期"])
pList = list(set(orignalData["分拣中心"].values))

i = 0
currentData: pd.DataFrame = orignalData[orignalData["分拣中心"] == pList[i]]
currentData = currentData.sort_values(["日期"])
ts = currentData["货量"]
ts.index = currentData["日期"]

# result = adfuller(ts.dropna())
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])

# plot_acf(ts, title="ACF")
# plot_pacf(ts, title="PACF")
# plt.show()

# 假设`ts`是一个Pandas Series格式的时间序列数据
model = auto_arima(ts, 
                   start_p=1, start_q=1,
                   test='adf',       # 使用adf测试来确定d的最佳值
                   max_p=10, max_q=10, # 最大的p和q
                   m=20,              # 季节性周期频率
                   d=None,           # 让模型自动确定d
                   seasonal=True,   # 是否考虑季节性
                   start_P=0, 
                   D=0, 
                   trace=True,       # 打印输出搜索过程
                   error_action='ignore',  
                   suppress_warnings=True, 
                   stepwise=True) 

