<h1>Book Popularity Predictor</h1>

Using Machine Learning techniques (Random Forest) we predict whether a book will become popular based on its metadata from the "Goodreads Books" dataset found on Kaggle: https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks/data 

<h3>How to Run:</h3>

<h4>1. Prerequisites</h4>

Ensure you have Python installed and your virtual environment active.

Run this command on your project's terminal:
pip install -r requirements.txt

<h4>2. Setup Credentials</h4>

Create a .env file in the root directory and add your Kaggle credentials.

The inside of the file must look like this:
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

<h4>3. Train the Model </h4>

Run the training script to fetch the data and save the model.

Run this command on your project's terminal:
python train_model.py

<h4>4. Make Predictions </h4>

Use the interactive script to test your own book ideas.

Run this command on your project's terminal:
python predict.py

<h3>Methodology:</h3>

Dataset: "Goodreads Books" dataset via Kaggle. <br>
Target Variable: is_popular (Binary classification based on the median number of ratings). <br>
Features Used: average_rating, num_pages, language_code and text_reviews_count. <br>
Model: RandomForestClassifier with 91% f1-score accuracy. <br>

<h3>Model Performance:</h3>

As shown in the Confusion Matrix (also found in the research_and_visuals.ipynb file), the model is well-balanced between predicting hits and average books: <br>
![Confusion Matrix](images/confusion_matrix.png)

<h3>Project Structure:</h3>

train_model.py: Automates data cleaning, encoding, and training. <br>
book_popularity_model.pkl: The saved machine learning model.<br>
predict.py: User interface for real-time predictions.<br>
research_and_visuals.ipynb: Data analysis and evaluation.<br>
.env: (Ignored by Git) Stores sensitive API keys.<br>
.gitignore: Keeps the repository clean of junk files.
