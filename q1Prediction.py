import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

orignalData = pd.read_csv("./data1.csv", encoding="gb2312", parse_dates=["日期"])
pList = list(set(orignalData["分拣中心"].values))

def predict(i: int):
    i = 0
    currentData: pd.DataFrame = orignalData[orignalData["分拣中心"] == pList[i]]
    currentData = currentData.sort_values(["日期"])
    ts = currentData["货量"]
    ts.index = currentData["日期"]

    p, d, q = 4, 2, 12
    _p, _d, _q, s = 1, 1, 1, 20
    model = sm.tsa.statespace.SARIMAX(ts,
                                    order=(p, d, q),
                                    seasonal_order=(_p, _d, _q, s),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    result = model.fit()
    predInSampleArima = result.predict(start=ts.index[0], end=ts.index[-1])

    a, b = np.polyfit(list(range(len(currentData))), currentData["货量"], 1)
    x = np.linspace(0, len(currentData), len(currentData))
    predInSampleLinear = a*x + b

    diffSumArima = 0
    diffSumLinear = 0
    for j in range(len(currentData)):
        diffSumArima += abs(predInSampleArima[i] - currentData.at[currentData.index[j], "货量"])
        diffSumLinear += abs(predInSampleLinear[i] - currentData.at[currentData.index[j], "货量"])
    diffSum = diffSumArima + diffSumLinear
    wArima = diffSumLinear / diffSum
    wLinear = diffSumArima / diffSum
    print(wArima)
    print(wLinear)
    
    plt.plot(currentData["日期"], predInSampleArima, label="Arima")
    plt.plot(currentData["日期"], predInSampleLinear, label="Linear")
    plt.plot(currentData["日期"], predInSampleArima * wArima + predInSampleLinear * wLinear, label="Combo")
    plt.plot(currentData["日期"], currentData["货量"], label="Orignal")
    plt.legend()
    plt.show()

predict(0)
