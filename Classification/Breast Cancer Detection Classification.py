#Breast Cancer Detection Classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Breast Cancer Detection Classification.csv')
X = data_set.iloc[:,1:-1]
y = data_set.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 21)

#Apply Kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 21)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))