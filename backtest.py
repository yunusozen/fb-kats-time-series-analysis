from kats.utils import backtesters
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pprint import pprint

import kats.utils.time_series_parameter_tuning as tpt
from kats.consts import ModelEnum, SearchMethodEnum
from kats.consts import TimeSeriesData
from kats.models.arima import ARIMAModel, ARIMAParams
from kats.models.prophet import ProphetModel, ProphetParams
from kats.utils.backtesters import BackTesterSimple

backtester_errors = {}


df = pd.read_csv("./data/air_passengers.csv")
df.columns = ["time", "value"]
# print(df.head())

ts = TimeSeriesData(df)

# ts.plot(cols=["value"])
# plt.show()
errors = ["mape", "smape", "mae", "mase", "mse", "rmse"]

params = ARIMAParams(p=2, d=1, q=1)

backtester_arima = BackTesterSimple(
    error_methods=errors,
    data=ts,
    params = params,
    train_percentage=75,
    test_percentage=25,
    model_class=ARIMAModel
)

backtester_arima.run_backtest()

backtester_errors["arima"] = {}

for error, value in backtester_arima.errors.items():
    backtester_errors["arima"][error] = value

pprint(backtester_errors["arima"])

params_prop = ProphetParams(seasonality_mode="multiplicative")

backtester_prop = BackTesterSimple(
    error_methods=errors,
    data=ts,
    params = params_prop,
    train_percentage=75,
    test_percentage=25,
    model_class=ProphetModel
)

backtester_prop.run_backtest()

backtester_errors["prophet"] = {}

for error, value in backtester_prop.errors.items():
    backtester_errors["prophet"][error] = value

pprint(backtester_errors["prophet"])

print(pd.DataFrame.from_dict(backtester_errors))
