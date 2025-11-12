"""
Train Collaborative Filtering Model using Implicit (No Compilation Needed)
FIXED VERSION - Corrected dimension issues
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def train_cf_model(min_ratings=20):
    """
    Train CF model using Implicit library (pre-built wheels for Windows)
    FIXED: Corrected dimension handling for predictions
    """
    
    print("="*60)
    print("🤖 COLLABORATIVE FILTERING MODEL TRAINING (Implicit ALS)")
    print("="*60)
    
    # Load ratings data
    ratings_file = 'data/crop_ratings.csv'
    
    if not os.path.exists(ratings_file):
        print(f"❌ Error: {ratings_file} not found")
        print("💡 Run 'python scripts/generate_initial_data.py' first")
        return False
    
    df_ratings = pd.read_csv(ratings_file)
    print(f"\n📊 Loaded {len(df_ratings)} ratings")
    
    # Check if we have enough ratings
    if len(df_ratings) < min_ratings:
        print(f"⚠️  Warning: Only {len(df_ratings)} ratings (need {min_ratings})")
        print("💡 Generate more data or lower min_ratings threshold")
        return False
    
    # Display statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Unique users: {df_ratings['user_id'].nunique()}")
    print(f"   Unique crops: {df_ratings['crop'].nunique()}")
    print(f"   Rating range: {df_ratings['rating'].min():.1f} - {df_ratings['rating'].max():.1f}")
    print(f"   Average rating: {df_ratings['rating'].mean():.2f}")
    
    # Prepare data for Implicit
    print(f"\n🔄 Preparing data for Implicit ALS...")
    
    # Create mappings
    user_ids = df_ratings['user_id'].unique()
    crop_names = df_ratings['crop'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    crop_to_idx = {crop: idx for idx, crop in enumerate(crop_names)}
    idx_to_crop = {idx: crop for crop, idx in crop_to_idx.items()}
    
    # Create sparse matrix (user-crop matrix)
    rows = df_ratings['user_id'].map(user_to_idx)
    cols = df_ratings['crop'].map(crop_to_idx)
    values = df_ratings['rating'].values
    
    sparse_matrix = csr_matrix((values, (rows, cols)), 
                              shape=(len(user_ids), len(crop_names)))
    
    print(f"   Sparse matrix shape: {sparse_matrix.shape}")
    print(f"   Matrix density: {(sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])):.4f}")
    
    # Split data for evaluation
    train_data, test_data = train_test_split(df_ratings, test_size=0.2, random_state=42)
    
    # Create train sparse matrix
    train_rows = train_data['user_id'].map(user_to_idx)
    train_cols = train_data['crop'].map(crop_to_idx)
    train_values = train_data['rating'].values
    
    train_matrix = csr_matrix((train_values, (train_rows, train_cols)), 
                             shape=(len(user_ids), len(crop_names)))
    
    print(f"\n🔄 Training Implicit ALS model...")
    
    # Initialize and train model
    model = implicit.als.AlternatingLeastSquares(
        factors=20,           # Number of latent factors (reduced for small dataset)
        regularization=0.1,   # Increased regularization for small dataset
        iterations=15,        # Number of training iterations
        random_state=42
    )
    
    # Implicit expects item-user matrix (transposed), so we use CSR format correctly
    # For explicit feedback, we use the confidence matrix directly
    model.fit(train_matrix)
    
    print("✅ Model trained successfully")
    
    # Simple evaluation - FIXED version
    print(f"\n📊 Model Evaluation:")
    
    # Calculate predictions for training data
    y_true = []
    y_pred = []
    
    for _, row in train_data.iterrows():
        user_idx = user_to_idx[row['user_id']]
        crop_idx = crop_to_idx[row['crop']]
        
        # Get the actual rating
        actual_rating = row['rating']
        y_true.append(actual_rating)
        
        # Predict using the model
        try:
            # Use the model's predict method
            predicted_score = model.predict(user_idx, crop_idx)
            y_pred.append(predicted_score)
        except:
            # Fallback: use dot product of user and item factors
            user_factor = model.user_factors[user_idx]
            item_factor = model.item_factors[crop_idx]
            predicted_score = np.dot(user_factor, item_factor)
            y_pred.append(predicted_score)
    
    # Calculate RMSE
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"   Training RMSE: {rmse:.4f}")
    
    # Calculate mean absolute error
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    print(f"   Training MAE: {mae:.4f}")
    
    # Save model with all necessary components
    model_package = {
        'model': model,
        'user_to_idx': user_to_idx,
        'crop_to_idx': crop_to_idx,
        'idx_to_user': {idx: user for user, idx in user_to_idx.items()},
        'idx_to_crop': idx_to_crop,
        'user_ids': user_ids,
        'crop_names': crop_names,
        'sparse_matrix': sparse_matrix,
        'rmse': rmse,
        'mae': mae
    }
    
    # Save model
    model_path = 'models/cf_model.pkl'
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model_package, model_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    
    # Test predictions
    print(f"\n🧪 Testing predictions...")
    test_samples = min(3, len(df_ratings))
    
    for i in range(test_samples):
        user_id = df_ratings['user_id'].iloc[i]
        crop = df_ratings['crop'].iloc[i]
        actual_rating = df_ratings['rating'].iloc[i]
        
        if user_id in user_to_idx and crop in crop_to_idx:
            user_idx = user_to_idx[user_id]
            crop_idx = crop_to_idx[crop]
            
            try:
                predicted_score = model.predict(user_idx, crop_idx)
                print(f"   User: {user_id}, Crop: {crop}")
                print(f"     Actual: {actual_rating:.1f}, Predicted: {predicted_score:.2f}")
            except Exception as e:
                print(f"   Prediction failed for {user_id}, {crop}: {e}")
    
    print("\n" + "="*60)
    print("🎉 CF MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📌 Next steps:")
    print(f"   1. Update hybrid_recommender.py with Implicit support")
    print(f"   2. Run Streamlit app: streamlit run app/streamlit_app.py")
    print(f"   3. Use existing user IDs to test personalization")
    
    return True

if __name__ == "__main__":
    success = train_cf_model(min_ratings=20)
    sys.exit(0 if success else 1)