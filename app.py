from flask import Flask, render_template, request
import joblib
import pandas as pd
from data_preprocessing import preprocess_data, additional_features

# Initialize Flask app
app = Flask(__name__)

# Load ML model & vectorizer
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
expected_features = joblib.load("models/additional_features.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    phishing_prob = None  # Default value to prevent errors
    is_phishing = None
    email_text = ""

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()

        if not email_text:
            return render_template("index.html", email_text=email_text, error="⚠️ Please enter an email!")

        # Preprocess Email
        processed_email = preprocess_data([email_text])
        processed_email_series = pd.Series(processed_email)
        additional_feats = additional_features(processed_email_series)

        # Ensure feature alignment
        if set(expected_features) != set(additional_feats.columns.tolist()):
            return "Error: Feature mismatch between training and testing data."

        # Vectorize email
        X_test_tfidf = vectorizer.transform(processed_email)
        X_combined = pd.DataFrame.sparse.from_spmatrix(X_test_tfidf)
        X_combined = pd.concat([X_combined, additional_feats.reset_index(drop=True)], axis=1)
        X_combined.columns = X_combined.columns.astype(str)

        # Make prediction
        phishing_prob = float(model.predict_proba(X_combined)[0][1])  # Ensure it's a float
        is_phishing = phishing_prob > 0.5  # Adjust threshold if needed

    return render_template("index.html", email_text=email_text, phishing_prob=phishing_prob, is_phishing=is_phishing)

if __name__ == "__main__":
    app.run(debug=True)
