import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kats.consts import TimeSeriesData
from kats.models.theta import ThetaModel, ThetaParams

df = pd.read_csv("data/air_passengers.csv")
df.columns = ["time", "value"]

ts = TimeSeriesData(df)
params = ThetaParams(m=12)
m = ThetaModel(data=ts, params=params)
m.fit()
fcst = m.predict(steps=30, alpha=0.2)
m.plot()
plt.show()