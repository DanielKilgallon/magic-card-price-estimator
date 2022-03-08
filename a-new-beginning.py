from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math

filesG = os.listdir('training-images/GreaterThanCutOff')
filesL = os.listdir('training-images/LessThanCutOff')
some_list = []
for file in filesG:
    some_list.append(int(math.ceil(float(file.split('~')[0]))))
for file in filesL:
    some_list.append(int(math.ceil(float(file.split('~')[0]))))

sorted_list = sorted(some_list)
sorted_counted = Counter(sorted_list)

range_length = list(range(max(some_list))) # Get the largest value to get the range.
data_series = {}

for i in range_length:
    data_series[i] = 0 # Initialize series so that we have a template and we just have to fill in the values.

for key, value in sorted_counted.items():
    data_series[key] = value

data_series = pd.Series(data_series)
x_values = data_series.index

# you can customize the limits of the x-axis
# plt.xlim(0, max(some_list))
plt.bar(x_values, data_series.values)

plt.show() 