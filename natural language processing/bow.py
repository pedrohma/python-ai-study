from sklearn.feature_extraction.text import CountVectorizer

Sentences = ['We are using the Bag of Word model', 'Bag of Word model is used for extracting the features.']

vectorizer = CountVectorizer()

features_text = vectorizer.fit_transform(Sentences).todense()

print(vectorizer.vocabulary_)
