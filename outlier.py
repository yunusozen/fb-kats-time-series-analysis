import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pprint import pprint
from kats.consts import TimeSeriesData
from kats.detectors.outlier import OutlierDetector

df = pd.read_csv("./data/air_passengers.csv")
df.columns = ["time", "value"]

df.loc[df.time == "1950-12-01", "value"] *= 5

df.loc[df.time == "1955-12-01", "value"] *= 5

ts = TimeSeriesData(df)



ts_detector = OutlierDetector(ts, "additive")

ts_detector.detector()

pprint(ts_detector.outliers)

ts_rem = ts_detector.remover(interpolate=False)  
ts_int = ts_detector.remover(interpolate=True) #try False as well

# print(ts_rem)
# ts_rem.plot(cols=["y_0"])
# plt.show()

fig, ax = plt.subplots(figsize=(20,8), nrows=1, ncols=2)
ts_rem.to_dataframe().plot(x="time", y="y_0", ax=ax[0])
ts_int.to_dataframe().plot(x="time", y="y_0", ax=ax[1])
plt.show()

