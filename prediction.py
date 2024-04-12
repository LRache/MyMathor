import torch
import pandas as pd
import numpy as np

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import random
import warnings

warnings.filterwarnings('ignore')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(99)
np.random.seed(99)
random.seed(99)

df = pd.read_csv("./data1.csv", encoding="gb2312")
print(set(df["分拣中心"].values))

oData = df[df["分拣中心"] == "SC63"]
oData.sort_values("日期")
oData["DateIndex"] = list(range(len(oData)))
plt.plot(oData["DateIndex"], oData["货量"])
plt.show()

scaler = MinMaxScaler()
train_use = scaler.fit_transform(oData["货量"].reshape(-1, 1))
