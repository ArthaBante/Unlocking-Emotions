import pandas as pd
import string
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt')

file_path = r"C:\Users\Dell\OneDrive - University of Hertfordshire\Unlocking_Emotions_NLP_Game\Unlocking_Emotions_NLP_Game\datasets\archive\sentiment labelled sentences\Training_test.csv"
df = pd.read_csv(file_path)

df.columns = ["review", "rating"]

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)  # Tokenize words
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return " ".join(tokens)
    return ""


df["cleaned_review"] = df["review"].apply(preprocess_text)


df["sentiment"] = df["rating"].map({1: "positive", 0: "negative"})


df[["cleaned_review", "sentiment"]].to_csv("cleaned_reviews.csv", index=False)

print("✅ Text data cleaned and saved successfully!")
