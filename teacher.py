import json
import os
import random
import string

import nltk
import numpy as np
from flask import Flask, jsonify, render_template, request
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Initialize the Flask application
app = Flask(__name__)

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words("english"))

# Load the dataset
with open("data/teacher.json", "r") as file:
    data = json.load(file)


# Function to preprocess text
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in string.punctuation and word not in stop_words
    ]
    return " ".join(tokens)


# Extracting data from teacher
X = []
y = []
for intent in data["teacher"]:
    for pattern in intent["patterns"]:
        X.append(preprocess_text(pattern))
        y.append(intent["tag"])

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(y)

# Balance the dataset
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

# Reshape X for oversampling
X = np.array(X).reshape(-1, 1)
X, y = ros.fit_resample(X, y)
X = X.flatten()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create the pipeline with TfidfVectorizer and SVC
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=1500, ngram_range=(1, 2))),
        ("svc", SVC(kernel="linear", probability=True)),
    ]
)

# Expanded hyperparameter tuning
param_grid = {
    "svc__C": [0.1, 1, 10, 50, 100],
    "tfidf__max_features": [None, 500, 1000, 1500],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
}

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearch with Cross-Validation
grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

# Print best parameters
print(f"Best Parameters: {grid.best_params_}")


# Define a function to get a response based on the predicted tag
def get_response(teacher, tag):
    for intent in teacher:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])


# Define a function to predict the response from the user's input
def chatbot_response(user_input):
    user_input = preprocess_text(user_input)
    predicted_tag_index = grid.best_estimator_.predict([user_input])[0]
    predicted_tag = le.inverse_transform([predicted_tag_index])[0]
    return get_response(data["teacher"], predicted_tag)


# Define route for home page
@app.route("/")
def home():
    return render_template("index.html")


# Define route for chatbot response
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["user_input"]
    response = chatbot_response(user_input)
    return jsonify({"response": response})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))  # Get port from environment or default to 5000
    app.run(host='0.0.0.0', port=port, debug=False)
