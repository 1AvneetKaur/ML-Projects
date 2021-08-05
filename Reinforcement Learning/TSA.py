#TSA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Thomson Sampling Facebook Ads Optimization.csv')

#Implementation
import math
import random
T = 15000
num_ads = 10
ads_selected = []
num_of_rewards_1 = [0] * num_ads
num_of_rewards_0 = [0] * num_ads
total_rewards = 0
for n in range(0,T):
    ad = 0
    max_random = 0
    for i in range(0, num_ads):
        random_beta = random.betavariate(num_of_rewards_1[i] + 1, num_of_rewards_0[i] + 1)
        if (random_beta > max_random):
          max_random = random_beta
          ad = i
        ads_selected.append(ad)
        reward = data_set.values[n,ad]
    
    if reward == 1:
        num_of_rewards_1[ad] = num_of_rewards_1[ad] + 1
    else:
        num_of_rewards_0[ad] = num_of_rewards_0[ad] + 1
    
    total_rewards =   total_rewards  + reward
    
#Visualising
plt.hist(ads_selected)
plt.title('Histogram of Ads selection')
plt.xlabel('Ads')
plt.ylabel('No. of ads selected')
plt.show()
    
        
        