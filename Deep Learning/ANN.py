# Deep Learning via ANN

#importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#importing data set
data_set = pd.read_csv('American Express User Exit Prediction.csv')
X = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values

# Encoding categorical data
#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

#One Hot Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X=sc.fit_transform(X)

y = y.reshape(len(y),1)
#splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#initialisation of input layer
ann = tf.keras.models.Sequential()

#initialisation of first hidden layer
ann.add(tf.keras.layers.Dense(units = 5, activation = 'relu'))

#Adding second hidden layer
ann.add(tf.keras.layers.Dense(units = 5, activation = 'relu'))

#Adding output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#Applying KNN

#Compiling ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training 
ann.fit(X_train, y_train, batch_size = 32, epochs = 120) #accuarcy of training set

#Prediction
print(ann.predict(sc.transform([[1.0,0.0,0.0,553,0,45,4,0,4,1,274150]]))>0.5)

y_pred = ann.predict(X_test)
y_pred = y_pred>0.5

#Confusion matrix and accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_pred, y_test)
print(cm)
print(accuracy_score(y_test,y_pred)) #accuracy of test set