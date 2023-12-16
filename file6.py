import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
c1,c2,c3,c4 = np.loadtxt('data.csv',unpack=True,delimiter = ',')
x= np.column_stack((c1,c3))
y= c4
clf = GaussianNB()
clf.fit(x,y)
predictions = clf.predict(x)

print(accuracy_score(y,predictions))