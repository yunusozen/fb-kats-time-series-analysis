from kats import detectors
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pprint import pprint
from kats.consts import TimeSeriesData

from kats.tsfeatures.tsfeatures import TsFeatures

df = pd.read_csv("./data/air_passengers.csv")
df.columns = ["time", "value"]

ts = TimeSeriesData(df)

model = TsFeatures()

output = model.transform(ts)

pprint(output)


