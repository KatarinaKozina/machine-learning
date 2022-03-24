import pandas as pd
import quandl

df= quandl.get('BATS/EDGA_GSQB') #quandl je baza podataka

print (df.head()) 
'''
What is data Head () in Python?
Slikovni rezultat za df.head() in python
DataFrame - head() function

The head() function is used to get the first n rows.
This function returns the first n rows for the object based on position.
It is useful for quickly testing if your object has the right type of data in it

'''
df= df[['Short Volume','Total Volume']]
df['HL_PCT']= (df['Short Volume']- df['Total Volume'])/ df['Total Volume']*100.0

df= df[['Short Volume','Total Volume', 'HL_PCT']]
print(df.head())
