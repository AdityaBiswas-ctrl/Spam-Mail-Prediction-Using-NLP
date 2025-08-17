def predict_spam_ham(text):
    """
    Predicts if a given text is spam or ham using the trained Logistic Regression model.

    Args:
        text (str): The email text to classify.

    Returns:
        tuple: A tuple containing the predicted label ('ham' or 'spam') and the probability of being spam.
    """
    # Clean the input text using the same function used for training data
    cleaned_text = clean_text(text)

    # Transform the cleaned text using the fitted TF-IDF vectorizer
    # Need to reshape for a single sample prediction
    text_vectorized = tfidf.transform([cleaned_text]).toarray()

    # Predict the label using the best logistic regression model
    prediction = grid_lr.predict(text_vectorized)[0]

    # Get the probability of the positive class (spam)
    probability = grid_lr.predict_proba(text_vectorized)[:, 1][0]

    # Map the numerical prediction back to 'ham' or 'spam'
    predicted_label = "spam" if prediction == 1 else "ham"

    return predicted_label, probability

# Example usage:
example_email_spam = "Subject: Claim your free prize now!"
example_email_ham = "Subject: Meeting tomorrow at 10 AM"

prediction_spam, prob_spam = predict_spam_ham(example_email_spam)
prediction_ham, prob_ham = predict_spam_ham(example_email_ham)

print(f"Email 1: '{example_email_spam}'")
print(f"Predicted: {prediction_spam}, Probability of Spam: {prob_spam:.4f}")

print(f"\nEmail 2: '{example_email_ham}'")
print(f"Predicted: {prediction_ham}, Probability of Spam: {prob_ham:.4f}")
