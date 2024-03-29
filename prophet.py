import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams

df = pd.read_csv("data/air_passengers.csv")
df.columns = ["time", "value"]

ts = TimeSeriesData(df)
params = ProphetParams(seasonality_mode="multiplicative")
m = ProphetModel(data=ts, params=params)
m.fit()
fcst = m.predict(steps=30, freq="MS")
m.plot()
plt.show()