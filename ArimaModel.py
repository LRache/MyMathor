import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

orignalData = pd.read_csv("./data1.csv", encoding="gb2312", parse_dates=["日期"])
pList = list(set(orignalData["分拣中心"].values))

i = 0
currentData: pd.DataFrame = orignalData[orignalData["分拣中心"] == pList[i]]
currentData = currentData.sort_values(["日期"])
ts = currentData["货量"]
ts.index = currentData["日期"]

def get_best():
    bestP = 0
    bestQ = 0
    bestAIC = 1000000000
    d = 2
    bestModel = None
    for p in range(2, 13):
        for q in range(2, 13):
            model = sm.tsa.statespace.SARIMAX(ts,
                                    order=(p, d, q),
                                    seasonal_order=(1, 1, 1, 20),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    
            aic = model.fit().aic
            if aic < bestAIC:
                bestAIC = aic
                bestP = p
                bestQ = q
                bestModel = model
    return bestP, bestQ, bestModel

p, d, q = 4, 2, 12
_p, _d, _q, s = 1, 1, 1, 20
model = sm.tsa.statespace.SARIMAX(ts,
                                  order=(p, d, q),
                                  seasonal_order=(_p, _d, _q, s),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
result = model.fit()

# p, q, model = get_best()
# result = model.fit()
# print(p, q)
# print(result.aic)
# print(result.bic)

# # 绘制残差图
# plt.figure(figsize=(10, 4))
# plt.plot(residuals)
# plt.title('Residuals')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.show()

# # 绘制残差的密度图
# plt.figure(figsize=(10, 4))
# residuals.plot(kind='kde')
# plt.title('Density of Residuals')
# plt.show()

pred_in_sample = result.predict(start=ts.index[0], end=ts.index[-1])
ts.plot(label="Orignal")
pred_in_sample.plot(label="Prediction")
plt.show()
