import quandl
import pandas as pd
import math
import numpy as np
from sklearn import preprosessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
print(df.head())
forecast_out = int(math.ceil(0.01 * len(df)))
print(df.head())
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())
'''
In our case, we've decided the features
are a bunch of the current values,
and the label shall be the price,
in the future, where the future is 1% of the
entire length of the dataset out.

The dropna() function is used to remove missing values.
Determine if rows or columns which contain missing values are removed.

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html
'''
