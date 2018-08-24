# Logistic Regression - measures the relationship between dependent variables and independent variables by estimating the probabilities using a logistic function

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.array([[2, 4.8], [2.9, 4.7], [2.5, 5], [3.2, 5.5], [6, 5], [7.6, 4],
              [3.2, 0.9], [2.9, 1.9],[2.4, 3.5], [0.5, 3.4], [1, 4], [0.9, 5.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# creating the logistic regression classifier

Classifier_LR = linear_model.LogisticRegression(solver = 'liblinear', C = 75)

# train this classifier

Classifier_LR.fit(X, y)

min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

mesh_step_size = 0.02

x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                 np.arange(min_y, max_y, mesh_step_size))

output = Classifier_LR.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)
plt.figure()
plt.pcolormesh(x_vals, y_vals, output, cmap = plt.cm.gray)

plt.scatter(X[:, 0], X[:, 1], c = y, s = 75, edgecolors = 'black', linewidth=1, cmap = plt.cm.Paired)

plt.xlim(x_vals.min(), x_vals.max())
plt.ylim(y_vals.min(), y_vals.max())
plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))
plt.show()
