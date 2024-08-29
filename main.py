import nltk # Natural Language Toolkit. 
from nltk.corpus import movie_reviews # Dataset

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download("movie_reviews") 

# Loading the dataset. Each tuple in the list will consist of a movie review and its corresponding category (positive or negative).
reviews = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

"""
for review in reviews: 
    print(review)
"""

df = pd.DataFrame(reviews, columns=["text", "sentiment"])

vect = CountVectorizer(max_features=2000)

# Transforming the text data into vectors, so that it can be used by the model.
review_tokens = vect.fit_transform(df["text"])
sentiments = df["sentiment"]

STATE = 50 # Arbitrary integer. Guarantees the dataset is consistently split in the same way every time.

# Splits the data into training and testing sets (20% for testing).
reviews_train, reviews_test, sentiment_train, sentiment_test = train_test_split(
    review_tokens, sentiments, test_size=0.2, random_state=STATE
)

# Naive Bayes classifier 
model = MultinomialNB()
model.fit(reviews_train, sentiment_train) # Trains the model

sentiment_prediction = model.predict(reviews_test) # Test the model with the test data

print(f"-Accuracy: {accuracy_score(sentiment_test, sentiment_prediction)}")
print(f"-Classification Report:\n{classification_report(sentiment_test, sentiment_prediction)}")

def predict_sentiment(text):
    return model.predict(vect.transform([text]))[0]


# Test
print(predict_sentiment("Fantastic movie!"))
print(predict_sentiment("It was a terrible. Dislike."))
print(predict_sentiment("The movie was okay, nothing special."))