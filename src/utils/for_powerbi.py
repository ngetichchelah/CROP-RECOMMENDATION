"""
Export data and predictions for Power BI dashboard
"""

import pandas as pd
import sqlite3
import joblib

def export_data_for_powerbi():
    """Export all necessary data for Power BI"""
    
    print("Exporting data for Power BI...")
    
    # 1. Load main dataset
    df = pd.read_csv('data/processed/crop_data_cleaned.csv')
    
    # 2. Load crop requirements
    crop_req = pd.read_csv('data/processed/crop_requirements_summary.csv')
    
    # 3. Connect to database
    conn = sqlite3.connect('data/database/crop_recommendation.db')
    
    # 4. Get model predictions for entire dataset
    model = joblib.load('models/crop_model_svm.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    
    X = df.drop('label', axis=1)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    df['predicted_crop'] = encoder.inverse_transform(predictions)
    df['confidence'] = (probabilities.max(axis=1) * 100).round(2)
    df['correct_prediction'] = (df['label'] == df['predicted_crop']).astype(int)
    
    # 5. Create model performance summary
    model_performance = pd.DataFrame({
        'Model': ['SVM', 'XGBoost', 'Random Forest', 'KNN'],
        'Accuracy': [0.9324, 0.9165, 0.9125, 0.9136],
        'Precision': [0.9334, 0.9174, 0.9140, 0.9156],
        'Recall': [0.9324, 0.9165, 0.9125, 0.9136],
        'F1_Score': [0.9326, 0.9169, 0.9130, 0.9140]
    })
    
    # 6. Create crop categories
    crop_categories = pd.DataFrame({
        'Crop': crop_req['label'],
        'Category': ['Fruit', 'Pulse', 'Pulse', 'Fruit', 'Cash Crop', 'Cash Crop',
                    'Fruit', 'Pulse', 'Cash Crop', 'Pulse', 'Pulse', 'Cereal',
                    'Fruit', 'Pulse', 'Pulse', 'Fruit', 'Fruit', 'Fruit',
                    'Pulse', 'Fruit', 'Cereal', 'Fruit'],
        'Economic_Value': ['Medium', 'Low', 'Low', 'High', 'Very High', 'High',
                          'High', 'Low', 'Medium', 'Low', 'Low', 'Medium',
                          'High', 'Low', 'Low', 'Medium', 'Medium', 'Medium',
                          'Low', 'High', 'Medium', 'Medium']
    })
    
    # 7. Save all files
    df.to_csv('data/processed/crops_with_predictions.csv', index=False)
    crop_req.to_csv('data/processed/crop_requirements.csv', index=False)
    model_performance.to_csv('data/processed/model_performance.csv', index=False)
    crop_categories.to_csv('data/processed/crop_categories.csv', index=False)
    
    print("Data exported successfully!")
    print(f"  - crops_with_predictions.csv ({len(df)} rows)")
    print(f"  - crop_requirements.csv ({len(crop_req)} rows)")
    print(f"  - model_performance.csv ({len(model_performance)} rows)")
    print(f"  - crop_categories.csv ({len(crop_categories)} rows)")
    
    conn.close()

if __name__ == "__main__":
    export_data_for_powerbi()