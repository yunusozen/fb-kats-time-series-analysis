from kats.models.ensemble.ensemble import BaseModelParams, EnsembleParams
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from kats.consts import TimeSeriesData

from kats.models.ensemble.kats_ensemble import KatsEnsemble

from kats.models import (
   arima,holtwinters,
   linear_model,
   prophet,
   quadratic_model,
   sarima,
   theta
)


df = pd.read_csv("./data/air_passengers.csv")
df.columns = ["time", "value"]
# print(df.head())

ts = TimeSeriesData(df)

# ts.plot(cols=["value"])
# plt.show()

params = EnsembleParams(
   [
      BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
      BaseModelParams("prophet", prophet.ProphetParams()),
      BaseModelParams("theta", theta.ThetaParams(m=12)),
      BaseModelParams("linear", linear_model.LinearModelParams())
   ]
)

kats_params = {
   "models": params,
   "aggregation": "median",
   "seasonality_length": 12,
   "decomposition_method": "multiplicative"
}

m = KatsEnsemble(data=ts, params=kats_params)
m.fit()

fcst = m.predict(steps=30)

m.aggregate()

m.plot()
plt.show()



