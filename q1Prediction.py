import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima
import csv

def auto_arima_model(ts: pd.Series) -> pmdarima.arima.ARIMA:
    model: pmdarima.arima.ARIMA = pmdarima.auto_arima(ts, stepwise=False, seasonal=True, m=20, n_jobs=-1)
    return model

def autoArimaPredictInSample(ts: pd.Series):
    return auto_arima_model(ts).predict_in_sample(start=1, end=len(ts))

def predictDaily():
    orignalData = pd.read_csv("./data1.csv", encoding="gb2312", parse_dates=["日期"])
    pList = list(set(orignalData["分拣中心"].values))
    result = pd.DataFrame(columns=["分拣中心", "日期", "货量"])
    for i in range(len(pList)):
        name = pList[i]
        currentData: pd.DataFrame = orignalData[orignalData["分拣中心"] == name]
        currentData = currentData.sort_values(["日期"])
        ts = currentData["货量"]
        ts.index = currentData["日期"]

        model = auto_arima_model(ts)
        print(model)
        predInSampleArima = model.predict_in_sample(start=1, end=len(ts))
        predArima = model.predict(30)

        # a, b = np.polyfit(list(range(len(currentData))), currentData["货量"], 1)
        # x = np.linspace(0, len(currentData), len(currentData))
        # predInSampleLinear = a*x + b

        # stdArima = np.var(predInSampleArima, ddof=1)
        # stdLinear = np.var(predInSampleLinear, ddof=1)
        # stdSum = stdArima + stdLinear
        # wArima = stdLinear / stdSum
        # wLinear = stdArima / stdSum
        # print(wArima)
        # print(wLinear)
        
        for date, pred in zip(pd.date_range(ts.index[-1], periods=30), predArima):
            result.loc[len(result.index)] = [name, date, int(pred)]
        # plt.clf()
        # plt.plot(currentData["日期"], predInSampleArima, label="Arima")
        # plt.plot(pd.date_range(ts.index[-1], periods=30), predArima, label="Prediction")
        # # plt.plot(currentData["日期"], predInSampleLinear, label="Linear")
        # # plt.plot(currentData["日期"], predInSampleArima * wArima + predInSampleLinear * wLinear, label="Combo")
        # plt.plot(currentData["日期"], currentData["货量"], label="Orignal")
        # plt.legend()
        # plt.savefig(f"./ComboPredictPlot/{name}.png")
    result.to_csv("./结果表1.csv", index=False)

def predictHourly():
    orignalData = pd.read_csv("./data2.csv", encoding="gb2312", parse_dates=["日期"])
    pList = list(set(orignalData["分拣中心"].values))
    for i in range(20):
        currentData: pd.DataFrame = orignalData[orignalData["分拣中心"] == pList[i]]
        currentData["Time"] = currentData["日期"] + pd.to_timedelta(currentData['小时'], unit='H')
        currentData = currentData.sort_values(["Time"])
        ts = currentData["货量"]
        ts.index = currentData["Time"]

        predInSampleArima = autoArimaPredictInSample(ts)

        a, b = np.polyfit(list(range(len(currentData))), currentData["货量"], 1)
        x = np.linspace(0, len(currentData), len(currentData))
        predInSampleLinear = a*x + b

        stdArima = np.var(predInSampleArima, ddof=1)
        stdLinear = np.var(predInSampleLinear, ddof=1)
        stdSum = stdArima + stdLinear
        wArima = stdLinear / stdSum
        wLinear = stdArima / stdSum
        print(wArima)
        print(wLinear)
        
        plt.clf()
        plt.plot(currentData["日期"], predInSampleArima, label="Arima")
        plt.plot(currentData["日期"], predInSampleLinear, label="Linear")
        plt.plot(currentData["日期"], predInSampleArima * wArima + predInSampleLinear * wLinear, label="Combo")
        plt.plot(currentData["日期"], currentData["货量"], label="Orignal")
        plt.legend()
        plt.savefig(f"./ComboPredictPlot/{pList[i]}.png")

predictDaily()
