import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kats.consts import TimeSeriesData

df = pd.read_csv("data/air_passengers.csv")
df.columns = ["time", "value"]
ts = TimeSeriesData(df)
ts.plot(cols=["value"])
plt.show()