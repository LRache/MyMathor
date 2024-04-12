import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data1.csv", encoding="gb2312")
pList = list(set(df["分拣中心"].values))

for name in pList:
    data = df[df["分拣中心"] == pList[0]]
    data.sort_values("日期")
    plt.clf()
    plt.plot(range(len(data)), data["货量"])
    plt.savefig(f"./PointPlot/{name}.png")
