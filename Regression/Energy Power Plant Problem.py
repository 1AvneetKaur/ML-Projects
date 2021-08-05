import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Energy Power Plant.csv')
X = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values

#Reshaping
y = y.reshape(len(y),1) #into 2d array

#feature scaling in range -3 to +3 from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Splitting dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

#training dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
a = r2_score(y_test, y_pred)
print(a)