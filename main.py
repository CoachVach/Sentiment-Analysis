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

for review in reviews: 
    print(review)