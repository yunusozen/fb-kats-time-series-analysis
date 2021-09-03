import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from kats.consts import TimeSeriesData
from kats.models.holtwinters import HoltWintersModel, HoltWintersParams

df = pd.read_csv("./data/air_passengers.csv")
df.columns = ["time", "value"]
# print(df.head())

ts = TimeSeriesData(df)

# ts.plot(cols=["value"])
# plt.show()

params = HoltWintersParams(
   trend="add",
   seasonal="mul",
   seasonal_periods=12
)

m = HoltWintersModel(data=ts, params=params)

m.fit()

fcst = m.predict(steps=30, alpha=0.1)

m.plot()
plt.show()



