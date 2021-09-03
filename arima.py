from kats.utils import backtesters
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

import kats.utils.time_series_parameter_tuning as tpt
from kats.consts import ModelEnum, SearchMethodEnum
from kats.consts import TimeSeriesData
from kats.models.arima import ARIMAModel, ARIMAParams



df = pd.read_csv("./data/air_passengers.csv")
df.columns = ["time", "value"]
# print(df.head())

ts = TimeSeriesData(df)

# ts.plot(cols=["value"])
# plt.show()

params_grid = [
    {
        "name":"p",
        "type":"choice",
        "values": list(range(1,3)),
        "value_type": "int",
        "is_ordered": True
    },
    {
        "name":"d",
        "type":"choice",
        "values": list(range(1,3)),
        "value_type": "int",
        "is_ordered": True
    },
    {
        "name":"q",
        "type":"choice",
        "values": list(range(1,3)),
        "value_type": "int",
        "is_ordered": True
    }
]

split = int(0.8*len(df))
train_ts = ts[0:split]
test_ts = ts[split:]

parameter_tuner_grid = tpt.SearchMethodFactory.create_search_method(
    objective_name="evaluation_metric",
    parameters=params_grid,
    selected_search_method=SearchMethodEnum.GRID_SEARCH
)

def eval_func(params):
    arima_params = ARIMAParams(p=params["p"],d=params["d"],q=params["q"])
    model = ARIMAModel(train_ts, arima_params)
    model.fit()
    model_pred = model.predict(len(test_ts))
    error = np.mean(np.abs(model_pred["fcst"].values - test_ts.value.values))
    return error

parameter_tuner_grid.generate_evaluate_new_parameter_values(
     evaluation_function=eval_func
 )

res = (
     parameter_tuner_grid.list_parameter_value_scores()
 )

print(res)


