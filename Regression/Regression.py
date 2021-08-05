# REGRESSION
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Salary.csv')
X = data_set.iloc[:,0:1].values
y = data_set.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)

#Training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Testing
y_pred = regressor.predict(X_test)

#Visualising training set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('years of experience VS salary', fontsize = 15, color = 'blue')
plt.xlabel('years of experience', color = 'blue', fontsize = 12)
plt.ylabel('salary', color = 'blue', fontsize = 14)
plt.show()

#Visualising test set
plt.scatter(X_test, y_test, color = 'red')  
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')  
plt.title('years of experience VS salary', fontsize = 15, color = 'blue')
plt.xlabel('years of experience', color = 'blue', fontsize = 12)
plt.ylabel('salary', color = 'blue', fontsize = 14)
plt.show()

