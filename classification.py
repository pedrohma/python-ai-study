from sklearn.datasets import load_breast_cancer

# Using the Breast Cancer Wisconsin Diagnostic Database (http://scikit-learn.org/stable/datasets/index.html#breast-cancer-wisconsin-diagnostic-database)

data = load_breast_cancer()

label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# ['malignant' 'benign']
print(label_names)

# mean radius
print(feature_names[0])

#  the first data instance is a malignant tumor the radius of which is 1.7990000e+01
print(features[0])
