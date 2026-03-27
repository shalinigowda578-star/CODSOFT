
# Step 1: Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 2: Load dataset
# Make sure train_data.txt is in same folder
df = pd.read_csv("train_data.txt", sep=":::", engine='python',
                 names=['id','title','genre','plot'])

# Step 3: Show data
print("First 5 rows:")
print(df.head())

# Step 4: Clean data
df = df[['genre', 'plot']]
df.dropna(inplace=True)

# Step 5: Show genre count (important for project)
print("\nGenre distribution:")
print(df['genre'].value_counts())

# Step 6: Convert text → numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['plot'])
y = df['genre']

# Step 7: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 8: Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 9: Predict
y_pred = model.predict(X_test)

# Step 10: Accuracy
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Step 11: Test with your own input
sample = ["A hero saves the world from aliens"]

sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

print("\nCustom Prediction:", prediction)