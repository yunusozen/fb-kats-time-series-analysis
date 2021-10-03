import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from kats.consts import TimeSeriesData
from kats.detectors.outlier import OutlierDetector

df = pd.read_csv("./data/air_passengers.csv")
df.columns = ["time", "value"]

df.loc[df.time == "1950-12-01", "value"] *= 3
df.loc[df.time == "1955-12-01", "value"] *= 3

ts = TimeSeriesData(df)
ts_detector = OutlierDetector(ts, "additive")
ts_detector.detector()
pprint(ts_detector.outliers)
ts_no_int = ts_detector.remover(interpolate=False)  
ts_int = ts_detector.remover(interpolate=True) 

fig, ax = plt.subplots(figsize=(10,15), nrows=3, ncols=1)
df.plot(x="time", y="value", ax = ax[0])
ts_no_int.to_dataframe().plot(x="time", y="y_0", ax=ax[1])
ts_int.to_dataframe().plot(x="time", y="y_0", ax=ax[2])
fig.tight_layout(pad=7.0)
plt.show()

