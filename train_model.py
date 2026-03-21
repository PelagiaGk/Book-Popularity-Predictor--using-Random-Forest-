import pandas as pd
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv() 

import kagglehub
from kagglehub import KaggleDatasetAdapter
import kaggle 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():

    #importing dataset
    kaggle.api.authenticate()
    books = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        handle="jealousleopard/goodreadsbooks",
        path="books.csv",
        pandas_kwargs={
            "on_bad_lines": "skip",  
            "encoding": "utf-8"     
        }
    )
    # Preprocessing
    books.columns = [col.strip() for col in books.columns]

    median_val = books['ratings_count'].median()
    books['is_popular'] = (books['ratings_count'] > median_val).astype(int)

    le = LabelEncoder()
    books['lang_encoded'] = le.fit_transform(books['language_code'])

    # Features and target
    X = books[['average_rating', 'num_pages', 'lang_encoded', 'text_reviews_count']]
    y = books['is_popular']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #Save the model
    joblib.dump(model, 'book_popularity_model.pkl')

if __name__ == "__main__":
    main()


