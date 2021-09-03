import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pprint import pprint

from kats.consts import TimeSeriesData
from kats.detectors.cusum_detection import CUSUMDetector

np.random.seed(100)

df = pd.DataFrame(
    {
        "time": pd.date_range("2021-01-01", periods=60),
        "increase": np.concatenate([np.random.normal(1,0.2,30), np.random.normal(2,0.2,30)])
    }
)

ts = TimeSeriesData(df.loc[:,["time", "increase"]])
# ts.plot(cols=["increase"])
# plt.show()

df_a = pd.read_csv("./data/air_passengers.csv")
df_a.columns = ["time", "value"]
# print(df.head())

ts_a = TimeSeriesData(df_a)

detector = CUSUMDetector(ts_a)
points = detector.detector(interest_window=[30,80])

detector.plot(points)
plt.show()
