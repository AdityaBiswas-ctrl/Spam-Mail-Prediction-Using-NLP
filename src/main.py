import pandas as pd
from src.preprocessing import clean_text
from src.features import get_tfidf_features
from src.models import train_and_evaluate

def main():
    # Load dataset
    df = pd.read_csv("data/spam_ham_dataset.csv")
    df["clean_text"] = df["text"].apply(clean_text)

    # Features + Labels
    X, y, vectorizer = get_tfidf_features(df["clean_text"], df["label_num"])

    # Train & Evaluate
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
