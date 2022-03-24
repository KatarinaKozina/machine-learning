import quandl
import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn



df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


forecast_col = 'Adj. Close' #Adj.close predvidjamo
df.fillna(value=-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df))) #uzimamo 1% podataka za prognozu-dobivamo broj
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out) #za broj(forecast out)
#pomicemo stupac forecast col prema gore, na zadnjih 35 mjesta bit ce NaN
#sto znaci da smo osigurali da uzimamo odredjeni broj podataka za predvidjanje
print(df['label'])
df.dropna(inplace=True) #izbacuje tih 35 NaN vrijednosti
print(df['label']) # printa novi dataframe bez zadnjih 35 vrijednosti

x= np.array(df.drop(['label'],axis=1)) #stvara niz bez label, axis =1, radi se o stupcu
y= np.array(df['label'])

x=preprocessing.scale(x) #Generally, you want your features in machine learning to be in a range of -1 to 1.
#This may do nothing, but it usually speeds up processing and can also help with accuracy.


#x= x[:-forecast_out+1]
df.dropna(inplace=True)
y=np.array(df['label'])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

clf = svm.SVR()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)
clf = LinearRegression()
clf = LinearRegression(n_jobs=-1)

for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)


