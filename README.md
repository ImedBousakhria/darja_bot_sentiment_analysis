# Darja Sentiment Analysis

This project aims to classify the sentiment of text written in **Darja** (Algerian Arabic) as either **positive** or **negative**. Using natural language processing (NLP) techniques, we have developed a machine learning model tailored to the unique linguistic features of Darja.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Performance](#model-and-performance)
- [Future Work](#future-work)
- [Contributors](#contributors)

---

## Overview

Sentiment analysis for **Darja**, a colloquial form of Arabic spoken in Algeria, presents unique challenges due to its informal structure, diverse vocabulary, and lack of standardized writing. This project tackles these challenges by building a model that identifies whether a Darja sentence expresses **positive** or **negative** sentiment.

## Dataset

The dataset used in this project consists exclusively of **Darja** sentences. Each sentence is labeled with a sentiment category:
- **Post:** A sentence written in Darja.
- **Polarity Class:** The sentiment of the sentence (`0` for negative, `1` for positive).

### Dataset Overview:
- The dataset is composed entirely of Darja text.
- Sentiment labels are binary, representing either **positive** or **negative**.

## Features

- **Sentiment Classification:** Classifies Darja text as either positive or negative.
- **Preprocessing:** Darja-specific text cleaning, tokenization, and feature extraction using `CountVectorizer`.
- **Model Tuning:** Hyperparameter optimization using `MLPClassifier` and `GridSearchCV`.



2. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- `nltk`
- `scikit-learn`
- `pandas`
- `numpy`

## Usage

1. **Data Preprocessing:**
   Preprocess the Darja text data before training the model.

   ```bash
   python preprocess.py
   ```

2. **Train the Model:**
   Train the sentiment analysis model using the Darja dataset.

   ```bash
   python train_model.py
   ```

3. **Predict Sentiment:**
   Use the trained model to predict the sentiment of a Darja sentence.

   ```bash
   python predict.py --input "your Darja sentence here"
   ```

## Model and Performance

- **Model:** `MLPClassifier` optimized with `GridSearchCV`.
- **Performance:** Achieved high accuracy on Darja-only data after model tuning.
- **Preprocessing Steps:**
  - Darja-specific text cleaning
  - Tokenization and feature extraction with `CountVectorizer`

## Future Work

- Expand the dataset to include more Darja sentences for better model generalization.
- Add a **neutral** sentiment category for more nuanced classification.
- Explore more advanced NLP models like **transformers** to improve overall accuracy.

## Contributors

- **Cheurfi Behadj Yassine**
- **Bousakhria Imed**

