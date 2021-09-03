from kats import detectors
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pprint import pprint
from kats.consts import TimeSeriesData
from kats.detectors.outlier import OutlierDetector
from kats.detectors.trend_mk import MKDetector

df = pd.read_csv("./data/air_passengers.csv")
df.columns = ["time", "value"]

ts = TimeSeriesData(df)

mk_detector = MKDetector(ts, threshold=.8)

tps = mk_detector.detector()
mk_detector.plot(tps)
plt.show()

