"""
Test script for the Complete CF Implementation
Includes the SimpleCF class definition to avoid serialization issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from collections import defaultdict
import joblib

# Include the SimpleCF class definition in the test script
class SimpleCF:
    """Simple Collaborative Filtering using user-based cosine similarity"""
    
    def __init__(self):
        self.user_ratings = defaultdict(dict)
        self.crop_ratings = defaultdict(dict)
        self.user_vectors = {}
        self.crop_index = {}
        self.user_similarities = {}
    
    def predict(self, user_id, crop, min_similarity=0.1):
        if user_id not in self.user_ratings:
            return None
            
        if crop in self.user_ratings[user_id]:
            return self.user_ratings[user_id][crop]
        
        similar_users = []
        for other_user, rating in self.crop_ratings.get(crop, {}).items():
            if other_user != user_id:
                similarity = self.user_similarities.get(user_id, {}).get(other_user, 0)
                if similarity >= min_similarity:
                    similar_users.append((similarity, rating))
        
        if not similar_users:
            return None
        
        total_similarity = sum(sim for sim, _ in similar_users)
        weighted_sum = sum(sim * rating for sim, rating in similar_users)
        predicted_rating = weighted_sum / total_similarity
        
        return max(1.0, min(5.0, predicted_rating))

def test_cf_complete():
    print("🧪 Testing Complete CF Implementation...")
    
    # Check if model exists
    model_path = 'models/cf_model.pkl'
    if not os.path.exists(model_path):
        print("❌ Model file not found. Please run train_cf_final.py first.")
        return False
    
    # Load ratings data for reference
    try:
        df_ratings = pd.read_csv('data/crop_ratings.csv')
        print(f"📊 Ratings data: {len(df_ratings)} ratings")
        print(f"👥 Users: {df_ratings['user_id'].unique().tolist()}")
        print(f"🌱 Crops: {df_ratings['crop'].unique().tolist()}")
    except Exception as e:
        print(f"❌ Error loading ratings: {e}")
        return False
    
    # Load model
    try:
        model_package = joblib.load(model_path)
        model = model_package['model_object']
        metadata = model_package['metadata']
        
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {model_package['model_type']}")
        print(f"   Users: {metadata['n_users']}, Crops: {metadata['n_crops']}")
        print(f"   Ratings: {metadata['n_ratings']}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test predictions
    print(f"\n🔍 Testing predictions:")
    
    # Test with known user-crop pairs
    test_cases = []
    for i in range(min(3, len(df_ratings))):
        user_id = df_ratings['user_id'].iloc[i]
        crop = df_ratings['crop'].iloc[i]
        actual_rating = df_ratings['rating'].iloc[i]
        test_cases.append((user_id, crop, actual_rating))
    
    # Add some edge cases
    test_cases.extend([
        ('FARMER_KE_001', 'maize', None),  # Should work if in data
        ('NEW_USER_999', 'rice', None),    # Should return None (new user)
        ('FARMER_KE_001', 'unknown_crop', None)  # Should return None
    ])
    
    for user_id, crop, actual_rating in test_cases:
        try:
            prediction = model.predict(user_id, crop)
            if prediction is not None:
                normalized = (prediction - 1) / 4.0  # Convert to 0-1 scale
                if actual_rating is not None:
                    print(f"   ✅ {user_id} + {crop}:")
                    print(f"      Actual: {actual_rating:.1f}, Predicted: {prediction:.2f}, Normalized: {normalized:.3f}")
                else:
                    print(f"   ✅ {user_id} + {crop}: {prediction:.2f} (normalized: {normalized:.3f})")
            else:
                print(f"   ⚠️  {user_id} + {crop}: No prediction available")
        except Exception as e:
            print(f"   ❌ {user_id} + {crop}: Error - {e}")
    
    print("\n🎉 CF Testing Complete!")
    return True

if __name__ == "__main__":
    test_cf_complete()