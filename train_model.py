import numpy as np
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import load_data, additional_features
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Step 1: Load and preprocess the data
logging.info("Loading and preprocessing data...")
emails, labels = load_data("data/phishing_emails.csv")  # Adjust the file path if necessary
logging.info("Data loaded successfully!")

# Step 2: Encode labels (0 for non-phishing, 1 for phishing)
logging.info("Encoding labels...")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Step 3: Generate additional features
logging.info("Generating additional features...")
additional_feats = additional_features(emails)

# Step 4: Initialize and fit the TfidfVectorizer
logging.info("Initializing and fitting TfidfVectorizer...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(emails)
logging.info(f"Vectorized email data with {X_tfidf.shape[1]} features.")

# Step 5: Combine TF-IDF features with additional features
logging.info("Combining features...")
X = pd.DataFrame.sparse.from_spmatrix(X_tfidf)
X = pd.concat([X, additional_feats.reset_index(drop=True)], axis=1)
X.columns = X.columns.astype(str)
logging.info(f"Combined feature data with {X.shape[1]} total features.")

# Step 6: Balance classes using SMOTE
logging.info("Balancing classes with SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, labels)

# Step 7: Split the dataset into training and testing sets
logging.info("Splitting the data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Step 8: Perform Randomized Search for hyperparameter tuning
logging.info("Performing Randomized Search...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_grid, n_iter=20, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

logging.info(f"Best Parameters: {random_search.best_params_}")

# Step 9: Train the best model
best_model = random_search.best_estimator_
logging.info("Training the best model...")
best_model.fit(X_train, y_train)

# Step 10: Evaluate the model
logging.info("Evaluating the model on the test set...")
y_pred = best_model.predict(X_test)
logging.info(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
logging.info("\n" + classification_report(y_test, y_pred))
logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Step 11: Save the trained model
logging.info("Saving the trained model and vectorizer...")
joblib.dump(best_model, 'models/phishing_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(additional_feats.columns.tolist(), 'models/additional_features.pkl')
logging.info("All components saved successfully.")
