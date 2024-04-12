import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

orignalData = pd.read_csv("./data1.csv", encoding="gb2312", parse_dates=["日期"])
pList = list(set(orignalData["分拣中心"].values))

i = 0
currentData: pd.DataFrame = orignalData[orignalData["分拣中心"] == pList[i]]
currentData = currentData.sort_values(["日期"])
a, b = np.polyfit(list(range(len(currentData))), currentData["货量"], 1)

x = np.linspace(0, len(currentData), len(currentData))
y = a * x + b
plt.plot(x, y)
plt.plot(list(range(len(currentData))), currentData["货量"])
plt.show()
