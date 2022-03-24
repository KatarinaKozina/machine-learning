import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')


#k=broj točaka za koje računamo udaljenosti
#predict, točka koju želimo predvidjeti
#data-ulazni podatci
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
#udaljenosti spremamo u niz        
    distances = []
    for group in data:
        for features in data[group]:
#racunamo udaljenosti pomocu NumPy biblioteke
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    print(distances)
    votes = [i[1] for i in sorted(distances)[:k]]
    print(sorted(distances))
    
    vote_result = Counter(votes).most_common(1)[0][0]
    print(vote_result)
   
    return vote_result

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# same as:
##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0],ii[1],s=100,color=i)
        
plt.scatter(new_features[0], new_features[1], s=100)

result = k_nearest_neighbors(dataset, new_features)
plt.scatter(new_features[0], new_features[1], s=100, color = result)  
plt.show()



