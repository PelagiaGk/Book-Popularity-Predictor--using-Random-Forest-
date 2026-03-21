import joblib
import pandas as pd

# 1. Load the trained model
# This is the "brain" we saved in the previous step
try:
    model = joblib.load('book_popularity_model.pkl')
    print("--- Model Loaded Successfully ---")
except FileNotFoundError:
    print("Error: 'book_popularity_model.pkl' not found. Run train_model.py first!")
    exit()

def get_prediction():
    print("\nEnter book details to see if it will be POPULAR:")
    
    try:
        # 2. Get user input
        rating = float(input("Average Rating (e.g. 4.2): "))
        pages = int(input("Number of Pages (e.g. 350): "))
        reviews = int(input("Number of Text Reviews (e.g. 100): "))
        
        # We'll assume English (0) for simplicity, or you can add input for it
        lang = 0 

        # 3. Format data for the model
        # The model expects: [['average_rating', 'num_pages', 'lang_encoded', 'text_reviews_count']]
        new_data = pd.DataFrame([[rating, pages, lang, reviews]], 
                                columns=['average_rating', 'num_pages', 'lang_encoded', 'text_reviews_count'])

        # 4. Make Prediction
        prediction = model.predict(new_data)
        probability = model.predict_proba(new_data)[0][1] # Chance of being in class 1

        print("-" * 30)
        if prediction[0] == 1:
            print(f"RESULT: This book is likely to be a HIT! (Confidence: {probability:.1%})")
        else:
            print(f"RESULT: This book will likely have AVERAGE popularity. (Confidence: {1-probability:.1%})")
        print("-" * 30)

    except ValueError:
        print("Please enter valid numbers!")

if __name__ == "__main__":
    while True:
        get_prediction()
        cont = input("Test another book? (y/n): ")
        if cont.lower() != 'y':
            break