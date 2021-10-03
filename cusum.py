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
        "value": np.concatenate([np.random.normal(1,0.2,30), np.random.normal(2,0.2,30)])
    }
)
ts = TimeSeriesData(df)

detector = CUSUMDetector(ts)
points = detector.detector()

plt.xticks(rotation=20)
detector.plot(points)
changepoint, metadata = points[0]
pprint(metadata.__dict__)
plt.show()
