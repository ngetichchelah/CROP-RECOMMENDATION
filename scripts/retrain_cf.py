"""
Retrain Collaborative Filtering model using Surprise SVD
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
import joblib
from utils.helpers import append_model_log, get_ratings_count

# Configuration
MIN_RATINGS_REQUIRED = 20
CF_MODEL_PATH = 'models/cf_model.pkl'

# SVD Hyperparameters
SVD_N_FACTORS = 15
SVD_N_EPOCHS = 30
SVD_LR_ALL = 0.005
SVD_REG_ALL = 0.02


def load_ratings_data():
    """Load ratings from CSV"""
    try:
        df = pd.read_csv('data/crop_ratings.csv')
        
        if df.empty:
            print("❌ No ratings found in data/crop_ratings.csv")
            return None
        
        print(f"✅ Loaded {len(df)} ratings")
        print(f"   - Unique users: {df['user_id'].nunique()}")
        print(f"   - Unique crops: {df['crop'].nunique()}")
        print(f"   - Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")
        
        return df
        
    except FileNotFoundError:
        print("❌ File not found: data/ratings.csv")
        return None
    except Exception as e:
        print(f"❌ Error loading ratings: {e}")
        return None


def train_cf_model(ratings_df):
    """
    Train SVD collaborative filtering model
    
    Parameters:
    -----------
    ratings_df : DataFrame with columns [user_id, crop, rating]
    
    Returns:
    --------
    trained_model, rmse
    """
    print()
    print("TRAINING COLLABORATIVE FILTERING MODEL (SVD)")
    print()
    
    # Prepare data for Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'crop', 'rating']], 
        reader
    )
    
    # Build full trainset
    trainset = data.build_full_trainset()
    
    # Initialize SVD algorithm
    algo = SVD(
        n_factors=SVD_N_FACTORS,
        n_epochs=SVD_N_EPOCHS,
        lr_all=SVD_LR_ALL,
        reg_all=SVD_REG_ALL,
        random_state=42
    )
    
    print(f"\nHyperparameters:")
    print(f"  - n_factors: {SVD_N_FACTORS}")
    print(f"  - n_epochs: {SVD_N_EPOCHS}")
    print(f"  - learning_rate: {SVD_LR_ALL}")
    print(f"  - regularization: {SVD_REG_ALL}")
    
    # Train model
    print("\nTraining...")
    algo.fit(trainset)
    print("✅ Training complete")
    
    # Evaluate with cross-validation
    print("\nCross-validation (5-fold)...")
    cv_results = cross_validate(
        algo, 
        data, 
        measures=['RMSE', 'MAE'], 
        cv=5, 
        verbose=False
    )
    
    mean_rmse = cv_results['test_rmse'].mean()
    mean_mae = cv_results['test_mae'].mean()
    
    print(f"✅ Cross-validation complete")
    print(f"   - RMSE: {mean_rmse:.4f}")
    print(f"   - MAE:  {mean_mae:.4f}")
    
    return algo, mean_rmse


def save_model(model, filepath):
    """Save trained model to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\n✅ Model saved: {filepath}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("CF MODEL RETRAINING SCRIPT")
    print("="*60)
    
    # Check if we have enough data
    n_ratings = get_ratings_count()
    print(f"\nCurrent ratings count: {n_ratings}")
    
    if n_ratings < MIN_RATINGS_REQUIRED:
        print(f"\n⚠️  Need at least {MIN_RATINGS_REQUIRED} ratings to train CF model")
        print(f"   Currently have: {n_ratings}")
        print(f"   Missing: {MIN_RATINGS_REQUIRED - n_ratings}")
        print("\n💡 Options:")
        print("   1. Collect more real farmer feedback")
        print("   2. Run 'python scripts/generate_demo_ratings.py' to create synthetic data")
        return
    
    # Load ratings
    ratings_df = load_ratings_data()
    
    if ratings_df is None:
        return
    
    # Sanity checks
    if ratings_df['rating'].min() < 1 or ratings_df['rating'].max() > 5:
        print("⚠️  Warning: Some ratings outside 1-5 range")
    
    # Train model
    try:
        model, rmse = train_cf_model(ratings_df)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return
    
    # Save model
    save_model(model, CF_MODEL_PATH)
    
    # Log training run
    append_model_log(
        model_type='collaborative_filtering',
        algorithm='SVD',
        n_ratings=len(ratings_df),
        rmse=rmse,
        filepath=CF_MODEL_PATH
    )
    
    print()
    print("SUCCESS! CF MODEL READY FOR HYBRID RECOMMENDATIONS")
    print()
    print(f"\nModel stats:")
    print(f"  - Training samples: {len(ratings_df)}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - Model path: {CF_MODEL_PATH}")
    print(f"\nNext steps:")
    print(f"  1. Restart your Streamlit app: streamlit run app/streamlit_app.py")
    print(f"  2. Hybrid recommendations will now be active!")


if __name__ == "__main__":
    main()