# POLYNOMIAL REGRESSION
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('RBC in Humans Polynomial Regression.csv')
X = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values

from sklearn.linear_model import LinearRegression #ultimately, we need to combine Polynomial and Linear Regression
from sklearn.preprocessing import PolynomialFeatures
regressor = PolynomialFeatures(degree = 7) 
X_pol = regressor.fit_transform(X)
linear_reg = LinearRegression()
linear_reg.fit(X_pol,y)

#Visualising
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color = 'red')
plt.plot(X,linear_reg.predict(X_pol), color = 'yellow') #predict is a feature of LinearRegression only
plt.xlabel('Age', color = 'blue', fontsize = 12)
plt.ylabel('RBC', color = 'blue', fontsize = 12)
plt.title('AGE VS RBC', color = 'black', fontsize = 15)
plt.show()

print(linear_reg.predict(regressor.fit_transform([[10]])))