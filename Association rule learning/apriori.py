#Apriori (Association rule learning)

#installing apriori in command line 

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
data_set = pd.read_csv('Big Basket.com Cart Apriori.csv', header = None)
transactions = [] #2d array
for i in range(0,7219):
    transactions.append([str(data_set.values[i,j]) for j in range(0,20)])

from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

#Visualising
results = list(rules)
print(results)

def inspect(results):
    product1 = [tuple(result[2][0][0])[0] for result in results]
    product2 = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(product1, product2, supports, confidences, lifts))
DataFrame_intelligence = pd.DataFrame(inspect(results), columns = ['product1', 'product2', 'Support', 'Confidence', 'Lift'])

print(DataFrame_intelligence)