#K-Means Clustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Amazon.com Clusturing Model.csv')
X = data_set.iloc[:,[2,4]].values

#find optimal no. of clusters via Elbow Method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('WCSS via Elbow Method', color = 'black', fontsize = 15)
plt.xlabel('No of Clusters', color = 'blue', fontsize = 13)
plt.ylabel('WCSS', color = 'blue', fontsize = 13)
plt.show()

#Training model 
kmeans =  KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_means = kmeans.fit_predict(X)
print(y_means)

#Visualise results
plt.scatter(X[y_means == 0,0],X[y_means == 0,1], s=100, c = 'cyan', label = 'cluster 1')
plt.scatter(X[y_means == 1,0],X[y_means == 1,1], s=100, c = 'yellow', label = 'cluster 2')
plt.scatter(X[y_means == 2,0],X[y_means == 2,1], s=100, c = 'pink', label = 'cluster 3')
plt.scatter(X[y_means == 3,0],X[y_means == 3,1], s=100, c = 'red', label = 'cluster 4')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s= 200, c = 'blue', label = 'centroids')
plt.title('K-Means Clustering', color = 'black', fontsize = 15)
plt.xlabel('Age', color = 'blue', fontsize = 15)
plt.ylabel('Rating', color = 'blue', fontsize = 15)
plt.legend()
plt.show()


