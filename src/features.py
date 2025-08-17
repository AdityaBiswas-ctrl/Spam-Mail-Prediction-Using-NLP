from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(texts, labels, max_features=3000):
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(texts).toarray()
    y = labels
    return X, y, tfidf
