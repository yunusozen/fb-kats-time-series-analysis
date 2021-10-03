import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pprint import pprint

from kats.consts import TimeSeriesData
from kats.detectors.bocpd import BOCPDetector
from kats.utils.simulator import Simulator

sim = Simulator(n=450, start="2021-01-01", freq="D")
ts = sim.level_shift_sim(noise=0.05, seasonal_period=1)

detector = BOCPDetector(ts)
points = detector.detector()

plt.xticks(rotation=20)
detector.plot(points)
plt.show()

changepoint, metadata = points[0]
pprint(metadata.__dict__)
