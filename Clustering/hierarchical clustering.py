#HIERARCHICAL CLUSTERING
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Amazon.com Clusturing Model.csv')
X = data_set.iloc[:,[2,4]].values

#optimal no of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendrogram', color = 'black', fontsize = 15)
plt.xlabel('Age', color ='blue', fontsize = 12)
plt.ylabel('Euclidean Distance', color ='blue', fontsize = 12)

#Training 
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage = 'ward')
yac = ac.fit_predict(X)
print(yac)

#Visualising results
plt.scatter(X[yac == 0,0], X[yac == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[yac == 1,0], X[yac == 1,1], s = 100, c = 'pink', label = 'Cluster 2')
plt.scatter(X[yac == 2,0], X[yac == 2,1], s = 100, c = 'yellow', label = 'Cluster 3')
plt.scatter(X[yac == 3,0], X[yac == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[yac == 4,0], X[yac == 4,1], s = 100, c = 'orange', label = 'Cluster 5')
plt.scatter(X[yac == 5,0], X[yac == 5,1], s = 100, c = 'magenta', label = 'Cluster 6')
plt.scatter(X[yac == 6,0], X[yac == 6,1], s = 100, c = 'green', label = 'Cluster 7')
plt.title('Age vs Rating', color = 'black', fontsize = 15)
plt.xlabel('Age', color = 'blue', fontsize = 12)
plt.ylabel('Rating', color = 'blue', fontsize = 12)
plt.legend()
plt.show()