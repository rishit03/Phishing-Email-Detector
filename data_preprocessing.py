import pandas as pd
import re
import numpy as np

# Function to generate additional features
def additional_features(emails, keywords=None):
    """Generate additional features from email content."""
    if isinstance(emails, list):
        emails = pd.Series(emails)

    features = pd.DataFrame()
    features['email_length'] = emails.apply(len)
    features['capital_count'] = emails.apply(lambda x: sum(1 for c in x if c.isupper()))
    features['digit_count'] = emails.apply(lambda x: sum(c.isdigit() for c in x))

    # Use default phishing keywords if none are provided
    if keywords is None:
        keywords = ['win', 'gift', 'prize', 'free', 'offer']
    for keyword in keywords:
        features[f'keyword_{keyword}'] = emails.apply(lambda x: len(re.findall(r'\b' + re.escape(keyword) + r'\b', x.lower())))

    return features

# Function to load and validate the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)

    # Ensure required columns exist
    required_columns = ['Email Text', 'Email Type']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"The dataset must contain columns: {required_columns}")

    # Handle missing values
    data = data.dropna(subset=required_columns)
    return data['Email Text'], data['Email Type']

# Function to clean email text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\@[\w]*', '', text)  # Remove email mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower().strip()  # Convert to lowercase
    return text

# Function to preprocess multiple emails
def preprocess_data(emails):
    return [clean_text(email) for email in emails]
