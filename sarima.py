import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from kats.consts import TimeSeriesData
from kats.models.sarima import SARIMAModel, SARIMAParams

df = pd.read_csv("./data/air_passengers.csv")
df.columns = ["time", "value"]
# print(df.head())

ts = TimeSeriesData(df)

# ts.plot(cols=["value"])
# plt.show()

params = SARIMAParams(
    p = 2,
    d=1,
    q=1,
    trend="ct",
    seasonal_order=(1,0,1,12)
)

m = SARIMAModel(data=ts, params=params)

m.fit()

fcst = m.predict(steps=30, freq="MS")

m.plot()
plt.show()



