#RANDOM FOREST REGRESSION
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Salary Data Support Vector regression.csv')
X = data_set.iloc[:,1].values
y = data_set.iloc[:,-1].values
 
#Reshaping
X = X.reshape(len(X),1)
y = y.reshape(len(y),1)

#training dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 12, random_state = 42)
regressor.fit(X,y)

a = regressor.predict([[10]])
print(a)

#smoothening the graph
X_grid = np.arange(min(X),max(X), 1)
X_grid = X_grid.reshape(len(X_grid),1)

#Visualising results
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid), color = 'yellow')
plt.title('Grade vs Salary', color = 'black', fontsize = 15)
plt.xlabel('Grade', color = 'blue', fontsize = 12)
plt.ylabel('Salary', color = 'blue', fontsize = 12)
plt.show()