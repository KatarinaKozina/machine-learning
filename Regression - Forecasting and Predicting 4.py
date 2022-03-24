import quandl
import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle


style.use('ggplot')

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True) #popunjava prazna mjesta

forecast_out = int(math.ceil(0.01 * len(df)))# ceil zaokruzuje na sljedecu vecu vrijednost, len vraća duljinu
#predvidjamo za 35 sljedecih dana
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
# forecast_out = 35,
#znaci da ce data frame forecast col biti pomaknut za 35 mjesta prema gore,
#znaci da ce se izgubiti prvih 35 vrijednosti a zadnjih 35 bit ce NaN
print(df['label'])

x = np.array(df.drop(['label'], axis=1))
#pravimo niz pomocu numpy biblioteke, a niz je isti kao i pocetni data frame ali bez stupca(1) label
# da je axis = 0, dataframe bi bio isti ali bez retka label
x = preprocessing.scale(x)

x_lately = x[-forecast_out:] #zadnjih 35 uzima

x = x[:-forecast_out] #zadnjih 35 ne uzima


df.dropna(inplace=True)

y= np.array(df['label'])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)

clf.fit(x_train, y_train)
#When you call fit method it estimates the best
#representative function for the the data points
#(could be a line, polynomial or discrete borders around).

confidence = clf.score(x_test, y_test)
print("confidence=")
print(confidence)


forecast_set = clf.predict(x_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


