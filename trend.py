from kats import detectors
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pprint import pprint

from kats.consts import TimeSeriesData
from kats.detectors.trend_mk import MKDetector
from kats.utils.simulator import Simulator

sim = Simulator(n=365, start="2021-01-01", freq="D")
ts = sim.trend_shift_sim(
    noise=200, 
    seasonal_period=7, 
    seasonal_magnitude=0.007, 
    cp_arr=[250], 
    intercept = 10000, 
    trend_arr=[40,-20]
    )

mk_detector = MKDetector(ts, threshold=.8)

points = mk_detector.detector(direction="down", window_size=30, freq="weekly")
mk_detector.plot(points)
plt.show()

changepoint, metadata = points[0]
pprint(metadata.__dict__)
