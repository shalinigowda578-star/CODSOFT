
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("train_data.txt", sep=":::", engine='python',
                 names=['id','title','genre','plot'])

print("First 5 rows:")
print(df.head())

df = df[['genre', 'plot']]
df.dropna(inplace=True)
print("\nGenre distribution:")
print(df['genre'].value_counts())
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['plot'])
y = df['genre']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

sample = ["A hero saves the world from aliens"]

sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

print("\nCustom Prediction:", prediction)