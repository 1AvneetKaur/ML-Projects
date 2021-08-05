#SUPPORT VECTOR REGRESSION
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Salary Data Support Vector regression.csv')
X = data_set.iloc[:,1].values
y = data_set.iloc[:,-1].values

#Feature Scaling
X = X.reshape(len(X),1)
y = y.reshape(len(y),1) #needs to be converted into 2D to perform feature scaling on it

#Training
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Applying SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#Predicting results
a= sc_y.inverse_transform(regressor.predict(sc_X.transform([[10]])))  #used inverse_tranform to inverse the transform
print(a)

#Visualising results
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)), color = 'yellow')
plt.title('SVR',color = 'black', fontsize = 15)
plt.xlabel('Grade', color = 'blue', fontsize = 12)
plt.ylabel('Salary', color = 'blue', fontsize = 12)
plt.show()