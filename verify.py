# Create a quick verification script: verify_setup.py
import os
import sys

required_files = {
    'scripts/hybrid_recommender.py': 'Hybrid recommender system',
    'scripts/retrain_cf.py': 'CF training script',
    'utils/helpers.py': 'Helper functions for logging',
    'app/streamlit_app.py': 'Streamlit app',
    'models/crop_model_svm.pkl': 'SVM model',
    'models/scaler.pkl': 'Scaler',
    'models/label_encoder.pkl': 'Label encoder',
    'data/processed/crop_data_cleaned.csv': 'Training data',
}

print("🔍 Verifying Setup...\n")
all_good = True

for file_path, description in required_files.items():
    exists = os.path.exists(file_path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {file_path}")
    if not exists:
        all_good = False

if all_good:
    print("✅ All required files exist!")
    print("Ready to proceed to Phase 2")
else:
    print("❌ Some files are missing. Please create them first.")
    sys.exit(1)