#!/usr/bin/env python3
import pandas as pd

print(pd.__version__)

data = pd.read_csv("./taxi_dataset/validation_data.csv", error_bad_lines=False)