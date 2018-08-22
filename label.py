import numpy as np
from sklearn import preprocessing

# Sample input labels
input_labels = ['red','black','red','green','black','yellow','white']

# Creating the label encoder
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# encoding a set of labels
encoded_values = encoder.transform(input_labels)
print("\nLabels =", input_labels)
print("Encoded values =", list(encoded_values))

# decoding a set of values
new_encoded_values = [0, 1]
decoded_list = encoder.inverse_transform(new_encoded_values)
print("\nDecoded labels =", list(decoded_list))
