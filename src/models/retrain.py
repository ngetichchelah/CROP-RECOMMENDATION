"""
Retrain SVM model with ONLY original 7 features (no engineered features)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import joblib

print()
print("RETRAINING SVM WITH 7 ORIGINAL FEATURES ONLY")
print()

# Load data
df = pd.read_csv('data/processed/crop_data_cleaned.csv')

# ONLY USE ORIGINAL 7 FEATURES (NO ENGINEERED FEATURES)
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]
y = df['label']

print(f"\n✅ Loaded data: {len(df)} samples")
print(f"✅ Features: {features}")
print(f"✅ Crops: {y.nunique()}")

# Train-test split (80-20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Train: {len(X_train)}, Test: {len(X_test)}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Label encoding
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

print(f"\n✅ Scaling complete (mean=0, std=1)")
print(f"✅ Encoding complete ({len(encoder.classes_)} classes)")

# Train SVM
print()
print("TRAINING SVM MODEL")
print()

svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True,
    random_state=42
)

svm_model.fit(X_train_scaled, y_train_encoded)

# Evaluate
train_acc = svm_model.score(X_train_scaled, y_train_encoded)
test_acc = svm_model.score(X_test_scaled, y_test_encoded)

print(f"\n✅ Training accuracy: {train_acc*100:.2f}%")
print(f"✅ Test accuracy: {test_acc*100:.2f}%")

# Cross-validation
cv_scores = cross_val_score(svm_model, X_train_scaled, y_train_encoded, cv=5)
print(f"✅ Cross-validation: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# Save models
print()
print("SAVING MODELS")
print()

joblib.dump(svm_model, 'models/crop_model_svm.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/label_encoder.pkl')

print("✅ Saved: models/crop_model_svm.pkl")
print("✅ Saved: models/scaler.pkl")
print("✅ Saved: models/label_encoder.pkl")

# Verify by loading and testing
print()
print("VERIFICATION")
print()

loaded_model = joblib.load('models/crop_model_svm.pkl')
loaded_scaler = joblib.load('models/scaler.pkl')
loaded_encoder = joblib.load('models/label_encoder.pkl')

# Test with sample
sample = X_test.iloc[0:1]
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)
crop_name = loaded_encoder.inverse_transform(prediction)[0]

print(f"✅ Test prediction: {crop_name}")
print(f"✅ Expected features: {list(sample.columns)}")
print(f"✅ Model ready for deployment!")

print()
print("SUCCESS! Model retrained with 7 features only")
print()