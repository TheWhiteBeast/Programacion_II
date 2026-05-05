# Glassdoor Reviews Analysis: NLP & MLOps Challenge

This repository contains the code and documentation for the Natural Language Processing (NLP) and MLOps challenge, part of the Master in Data Science (MCD) program. The goal of this project is to extract, process, and classify employee feedback from Glassdoor using machine learning, and track the entire lifecycle using MLflow.

## 1. Dataset Extraction (Web Scraping)
The initial stage of this project involved gathering raw data from Glassdoor. The extraction pipeline reads the web pages, extracts specific content such as the company name, employee reviews, pros, and cons, and compiles all this information into a structured Pandas DataFrame. The resulting dataset serves as the foundation for our corpus analysis.

## 2. Text Preprocessing & Model Construction
The core of the NLP pipeline is divided into preprocessing and classification:

* **Text Preprocessing:** The text data underwent rigorous cleaning. We created a targeted corpus and applied several normalization steps: lowercasing, punctuation removal, stop-words filtering, and lemmatization using the NLTK library. We also calculated and plotted N-gram distributions to understand the most frequent word combinations.
* **Sentiment Analysis:** We utilized VADER (Valence Aware Dictionary and sEntiment Reasoner) to establish an initial ground truth for sentiment (Positive, Negative, Neutral) based on the text's polarity.
* **Model Construction:** To classify the text, we extracted the main features by calculating grammatical probabilities using TF-IDF (Term Frequency-Inverse Document Frequency). We then trained a Multinomial Naive Bayes model to predict the sentiment of new reviews based on these features.

## 3. MLOps Pipeline
To ensure best practices and reproducibility, we integrated MLflow to manage our machine learning lifecycle. The MLOps pipeline automatically performs the following tasks:
* Logs parameters (e.g., model type, vectorizer, max features).
* Logs evaluation metrics (Accuracy, Precision, Recall, F1-Score).
* Saves and logs artifacts, including the confusion matrix plot, N-grams distribution plot, and the TF-IDF vectorizer.
* Registers the model signatures (input/output schema) and tracks the run locally.

## 4. Execution Guide
To run this solution locally and reproduce the results, follow these technical instructions:

**Step 1: Pull the code from GitHub**
Open your terminal and clone this repository:
```bash
git clone <your_github_repository_url>
cd <your_repository_folder_name>