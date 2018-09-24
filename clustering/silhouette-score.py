import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples = 500, centers = 4, cluster_std = 0.40, random_state = 0)

scores = []
values = np.arange(2, 10)

# train it with the input data
for num_clusters in values:
    kmeans = KMeans(init = 'k-means++', n_clusters = num_clusters, n_init = 10)
    kmeans.fit(X)

# estimate the silhouette score for the current clustering model using the Euclidean distance metric
score = silhouette_score(X, kmeans.labels_, metric = 'euclidean', sample_size = len(X))

print("Number of clusters =", num_clusters)
print("Silhouette score =", score)
scores.append(score)

num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters =', num_clusters)
