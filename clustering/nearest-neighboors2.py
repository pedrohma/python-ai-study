from sklearn.datasets import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

# display the image of digit to verify what image we have to test

def Image_display(i):
   plt.imshow(digit['images'][i],cmap = 'Greys_r')
   plt.show()

# Actually there are total 1797 images but we are using the first 1600 images as training sample and the remaining 197 would be kept for testing purpose.

digit = load_digits()
digit_d = pd.DataFrame(digit['data'][0:1600])

train_x = digit['data'][:1600]
train_y = digit['target'][:1600]
KNN = KNeighborsClassifier(20)
KNN.fit(train_x,train_y)

KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski',
   metric_params = None, n_jobs = 1, n_neighbors = 20, p = 2,
   weights = 'uniform')

test = np.array(digit['data'][1725])
test1 = test.reshape(1,-1)
Image_display(1725)

KNN.predict(test1)

digit['target_names']
