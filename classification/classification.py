from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

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

#train_test_split function from sklearn and the command below will split the data into training and test data. In the example given below, we are using 40 % of the data for testing and the remaining data would be used for training the model

train, test, train_labels, test_labels = train_test_split(features,labels,test_size = 0.40, random_state = 42)

print(test_labels)

gnb = GaussianNB()

# training the model by fitting it to the data

model = gnb.fit(train, train_labels)

# evaluate the model by making predictions on our test data. 0s and 1s are the predicted values for the tumor classes – malignant and benign.

preds = gnb.predict(test)

print(preds)

# comparing the two arrays namely test_labels and preds, we can find out the accuracy of our model

print(accuracy_score(test_labels,preds))
