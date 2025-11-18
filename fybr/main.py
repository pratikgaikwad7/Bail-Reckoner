import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving and loading the model

# Load the dataset
data = pd.read_csv('bail_eligibility_dataset.csv')

# --- 1️⃣ Exploratory Data Analysis (EDA) ---
print("Dataset Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

print("\nFirst 5 Records:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='eligible_for_bail', data=data)
plt.title('Bail Eligibility Distribution')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# --- 2️⃣ Data Preprocessing & Model Training ---

# Encode categorical variables
label_encoders = {}
for column in ['offence', 'residential_stability', 'prior_record']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and Target
X = data.drop('eligible_for_bail', axis=1)
y = data['eligible_for_bail']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --- 3️⃣ Model Evaluation ---
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_rep)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- 4️⃣ Save the Model as a .pkl File ---
model_filename = 'bail_eligibility_model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved as '{model_filename}'")

# --- 5️⃣ Load the Model and Make Predictions ---
# Load the model
loaded_model = joblib.load(model_filename)
print("\nModel loaded successfully!")

# Function to predict bail eligibility using the loaded model
def predict_bail(ipc_section, offence, residential_stability, prior_record):
    # Encode the input
    offence_encoded = label_encoders['offence'].transform([offence])[0]
    residential_stability_encoded = label_encoders['residential_stability'].transform([residential_stability])[0]
    prior_record_encoded = label_encoders['prior_record'].transform([prior_record])[0]

    # Prepare the input array
    input_data = pd.DataFrame({
        'ipc_section': [ipc_section],
        'offence': [offence_encoded],
        'residential_stability': [residential_stability_encoded],
        'prior_record': [prior_record_encoded]
    })

    # Make the prediction using the loaded model
    prediction = loaded_model.predict(input_data)[0]
    return "Bail Approved" if prediction == 1 else "Bail Not Approved"

# Example User Input
ipc_section = 307
offence = 'Assault'
residential_stability = 'Stable'
prior_record = 'No'

result = predict_bail(ipc_section, offence, residential_stability, prior_record)
print("\nPrediction Result:")
print(result)
