# Sentiment-Analysis

## Installation

Setup - Install necessary libraries (if not installed):

```bash
pip install nltk pandas scikit-learn
```
## Summary

This project focuses on sentiment analysis within the field of natural language processing (NLP). Using Python libraries like NLTK and Scikit-learn, the project uses the movie_reviews dataset to build a model that can classify movie reviews as positive or negative. The workflow involves loading and preprocessing text data and training a Multinomial Naive Bayes classifier. After training, the model's performance is evaluated through accuracy and classification metrics. This project seeks to illustrate practical uses of sentiment analysis in text data.

## Dataset

The dataset used in this project consists of 2,000 movie reviews labeled as either positive or negative. Each review in the dataset is a text document where the sentiment has been pre-assigned, making it ideal for training and evaluating sentiment analysis models. The dataset is split evenly between positive and negative reviews, so it is ideal for building and testing a classification model. 

## Data processing

The program processes the movie_reviews dataset by reading the reviews and their associated sentiment labels from the dataset.Then it transforms this text data into numerical features using the CountVectorizer (provided by SciKit Learn). This tool tokenizes the text, creating a feature matrix where each row represents a review and each column corresponds to a unique word from the dataset, with values indicating word frequencies. The transformed data is then split into training and testing sets.

## Multinomial Naive Bayes Model

We use a Naive Bayes model, which is a probabilistic classifier that works well for situations where features are discrete, such as word counts in text classification tasks. It assumes that the words are conditionally independent given the class label, which simplifies the computation of probabilities. It calculates the likelihood of each word occurring within a particular class (positive or negative sentiment) and uses these probabilities to predict the class of a new text. 

## Evaluation of the Model

After training the model, we evaluate its performance by using it to predict sentiments for the reviews in the test dataset. This test data was not used during training, so it is new to the model, making it an effective measure for assessing how well the model generalizes to unseen data.

The program generates a report that indicates the model’s accuracy, precision, recall, and F1-score. 
- Accuracy measures the overall correctness of the model's predictions by calculating the proportion of correct predictions out of all predictions made. 
- Precision assesses the accuracy of the positive predictions, showing how many of the predicted positive instances are actually positive. 
- Recall evaluates the model’s ability to identify all relevant positive instances, indicating how many of the actual positive cases were correctly identified by the model. 
- F1-score provides a balanced measure that combines precision and recall into a single metric, useful for evaluating the model's performance when there is a need to balance both precision and recall.



### Author: Erik Kovach