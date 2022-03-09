from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math

# Get prices from files
filesG = os.listdir('all-images/')
price_data = []
for file in filesG:
    price_data.append(float(file.split('~')[0]))

# Modify price data
modified_price_data = []
for price_point in price_data:
    modified_price_data.append(np.log(price_point) / np.log(10))

sorted_list = sorted(modified_price_data)
sorted_counted = Counter(sorted_list)

range_length = list(range(int(max(modified_price_data)))) # Get the largest value to get the range.
data_series = {}

for i in range_length:
    data_series[i] = 0 # Initialize series so that we have a template and we just have to fill in the values.

for key, value in sorted_counted.items():
    data_series[key] = value

data_series = pd.Series(data_series)
x_values = data_series.index

# you can customize the limits of the x-axis
# plt.xlim(0, max(modified_price_data))
plt.bar(x_values, data_series.values, width=0.1)

plt.savefig('example.png')