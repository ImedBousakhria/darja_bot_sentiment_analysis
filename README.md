# Darja Sentiment Analysis

This project aims to classify the sentiment of text written in **Darja** (Algerian Arabic) as either **positive** or **negative**. Using natural language processing (NLP) techniques, we have developed a machine learning model tailored to the unique linguistic features of Darja.



## Overview

Sentiment analysis for **Darja**, a colloquial form of Arabic spoken in Algeria, presents unique challenges due to its informal structure, diverse vocabulary, and lack of standardized writing. This project tackles these challenges by building a model that identifies whether a Darja sentence expresses **positive** or **negative** sentiment.

## Dataset

The dataset used in this project consists exclusively of **Darja** sentences. Each sentence is labeled with a sentiment category:
- **Post:** A sentence written in Darja.
- **Polarity Class:** The sentiment of the sentence (`0` for negative, `1` for positive).

## Features

- **Sentiment Classification:** Classifies Darja text as either positive or negative.
- **Preprocessing:** Darja-specific text cleaning, tokenization, and feature extraction using `CountVectorizer`.
- **Model Tuning:** Hyperparameter optimization using `MLPClassifier` and `GridSearchCV`.



## Model and Performance

- **Model:** `MLPClassifier` 
- **Performance:** Achieved high accuracy on Darja-only data after model tuning.
- **Preprocessing Steps:**
  - Darja-specific text cleaning
  - Tokenization and feature extraction with `CountVectorizer`

## Future Work

- Expand the dataset to include more Darja sentences for better model generalization.
- Explore more advanced NLP models like **transformers** to improve overall accuracy.
- use better NER models 

## Contributors

- **Bousakhria Imed**
- **Cheurfi Behadj Yassine**
  

