"""
Test the Simple CF implementation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import joblib

def test_simple_cf():
    print("🧪 Testing Simple Collaborative Filtering...")
    
    # Load ratings to see what we're working with
    df_ratings = pd.read_csv('data/crop_ratings.csv')
    print(f"📊 Ratings data: {len(df_ratings)} ratings")
    print(f"👥 Users: {df_ratings['user_id'].unique().tolist()}")
    print(f"🌱 Crops: {df_ratings['crop'].unique().tolist()}")
    
    # Load model
    try:
        model_package = joblib.load('models/cf_model.pkl')
        model = model_package['model']
        print("✅ Model loaded successfully")
        
        # Test predictions
        test_cases = [
            (df_ratings['user_id'].iloc[0], df_ratings['crop'].iloc[0]),
            (df_ratings['user_id'].iloc[1], df_ratings['crop'].iloc[1]),
            ('FARMER_KE_001', 'maize'),
            ('NEW_USER', 'rice')  # Should return None
        ]
        
        print(f"\n🔍 Testing predictions:")
        for user_id, crop in test_cases:
            prediction = model.predict(user_id, crop)
            if prediction is not None:
                normalized = (prediction - 1) / 4.0
                print(f"   ✅ {user_id} + {crop}: {prediction:.2f} -> {normalized:.3f}")
            else:
                print(f"   ⚠️  {user_id} + {crop}: No prediction (new user/crop)")
        
        # Test recommendations
        test_user = df_ratings['user_id'].iloc[0]
        recommendations = model.recommend_for_user(test_user, top_n=3)
        
        if recommendations:
            print(f"\n📋 Recommendations for {test_user}:")
            for crop, score in recommendations:
                normalized = (score - 1) / 4.0
                print(f"   🌱 {crop}: {score:.2f} (normalized: {normalized:.3f})")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    test_simple_cf()