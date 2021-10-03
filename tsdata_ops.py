import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kats.consts import TimeSeriesData

df = pd.read_csv("data/air_passengers.csv")
df.columns = ["time", "value"]
ts = TimeSeriesData(df)
ts_from_series = TimeSeriesData(time=df.time, value=df.value)

ts_1 = ts[0:2]
ts_2 = ts[2:5]
ts_1.extend(ts_2)

print(ts[1:5])
print(ts[1:5] + ts[1:5])
print(ts == ts_from_series)
print(ts_1)