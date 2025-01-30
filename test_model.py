import joblib
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data, additional_features

# Load the saved model and vectorizer
try:
    model = joblib.load('models/phishing_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    expected_features = joblib.load('models/additional_features.pkl')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure the model and vectorizer files are available.")
    exit()

# Example test email
test_email = "Urgent! Click the below link to claim!"

# Preprocess the email
processed_email = preprocess_data([test_email])
processed_email_series = pd.Series(processed_email)

# Generate additional features
additional_feats = additional_features(processed_email_series)

# Check for feature mismatch
if set(expected_features) != set(additional_feats.columns.tolist()):
    raise ValueError("Feature mismatch between training and testing data.")

# Vectorize the processed email
X_test_tfidf = vectorizer.transform(processed_email)

# Combine TF-IDF features with additional features
X_combined = pd.DataFrame.sparse.from_spmatrix(X_test_tfidf)
X_combined = pd.concat([X_combined, additional_feats.reset_index(drop=True)], axis=1)

# Convert all column names to strings
X_combined.columns = X_combined.columns.astype(str)

# Make a prediction
prediction = model.predict(X_combined)
probabilities = model.predict_proba(X_combined)
phishing_prob = probabilities[0][1]

# Output the prediction and probability
print(f"Phishing Probability: {phishing_prob:.2f}")
if prediction[0] == 1:
    print("The email is phishing!")
else:
    print("The email is not phishing.")
