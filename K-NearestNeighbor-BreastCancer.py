import numpy as np
import sklearn
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
print(df.head(10))
df.replace('?',-99999, inplace=True)
#inplace: bool, default False If False, return a copy.
#Otherwise, do operation inplace and return None.

df.drop(['id'],axis=1, inplace=True)


X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

print(X)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

#Define the classifier:
clf = neighbors.KNeighborsClassifier()

#train the classifier
clf.fit(X_train, y_train)

#test the classifier
accuracy = clf.score(X_test, y_test)
print(accuracy)


#example_measures = np.array([4,2,1,1,1,2,3,2,1])
#example_measures = example_measures.reshape(1, -1)
#print(example_measures)
#prediction = clf.predict(example_measures)
#print(prediction)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(2, -1)
#print(example_measures)
prediction = clf.predict(example_measures)
print(prediction)

#onoliko redova koliko ima sampleova, a kolumni (-1) što znači da će ih samo rasporedit
# a ne preostaje nisšta drugo nego da ima onda jednu kolumnu!


