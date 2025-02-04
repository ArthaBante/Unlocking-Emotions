from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import joblib

dataset= r"C:\Users\Dell\OneDrive - University of Hertfordshire\Unlocking_Emotions_NLP_Game\Unlocking_Emotions_NLP_Game\cleaned_reviews.csv"

df = pd.read_csv(dataset)

# Ensure correct column names
df.columns = ["cleaned_text", "emotion"]

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["emotion"]

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X, y)

# Save the model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Sentiment analysis model trained and saved successfully!")