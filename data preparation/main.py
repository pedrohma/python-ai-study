import numpy as np
from sklearn import preprocessing

# Defining sample data

input_data = np.array([[2.1, -1.9, 5.5],
                      [-1.5, 2.4, 3.5],
                      [0.5, -7.9, 5.6],
                      [5.9, 2.3, -5.8]])

# Preprocessing data (Binarization)

data_binarized = preprocessing.Binarizer(threshold = 0.5).transform(input_data)
print("\nBinarized data:\n", data_binarized)

# Mean Removal - eliminating the mean from feature vector so every feature is centered on zero.

data_scaled = preprocessing.scale(input_data)
print("\nMean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis = 0))

# Scaling - the values of every feature can vary between many random values, it avoid any feature to be synthetically large or small

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print ("\nMin max scaled data:\n", data_scaled_minmax)

# Normalization - modify the feature vectors
# L1 - Least Absolute Deviations - This kind of normalization modifies the values so that the sum of the absolute values is always up to 1 in each row

data_normalized_l1 = preprocessing.normalize(input_data, norm = 'l1')
print("\nL1 normalized data:\n", data_normalized_l1)

# L2 - Least Squares - This kind of normalization modifies the values so that the sum of the squares is always up to 1 in each row.

data_normalized_l2 = preprocessing.normalize(input_data, norm = 'l2')
print("\nL2 normalized data:\n", data_normalized_l2)
