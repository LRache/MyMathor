import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pmdarima

orignalData = pd.read_csv("./data1.csv", encoding="gb2312", parse_dates=["日期"])
pList = list(set(orignalData["分拣中心"].values))

for i in range(1):
    currentData: pd.DataFrame = orignalData[orignalData["分拣中心"] == pList[i]]
    ts = currentData["货量"]
    ts.index = currentData["Time"]
    print(ts)
    model: pmdarima.arima.ARIMA = pmdarima.auto_arima(ts, start_p=1, d=None, start_q=1, max_p=16, max_q=16, start_P=1, max_P=4, start_Q=1, max_Q=4, m=20, seasonal=True)
    predictInSample = model.predict_in_sample(start=1, end=len(currentData))
    plt.clf()
    plt.plot(ts.index, predictInSample, label="Prediction")
    ts.plot(label="Orignal")
    plt.legend()
    plt.savefig(f"./PredictPlot/{pList[i]}.png")
