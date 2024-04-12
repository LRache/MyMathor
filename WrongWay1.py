"""
对数据进行级比检验
"""

import pandas as pd
import numpy as np

df = pd.read_csv("./data1.csv", encoding="gb2312")

pList = list(set(df["分拣中心"].values))
data = df[df["分拣中心"] == pList[0]]
data.sort_values("日期")
cumsum = np.cumsum(data["货量"].values)
print(cumsum)
n = len(data)
lower_bound = np.exp(-2 / (n + 1))
upper_bound = np.exp(2 / (n + 1))
print(lower_bound, upper_bound)

for i in range(1, len(cumsum)):
    k = cumsum[i] / cumsum[i-1]
    if not (lower_bound <= k and k <= upper_bound):
        raise ValueError(k)
print("OK")
