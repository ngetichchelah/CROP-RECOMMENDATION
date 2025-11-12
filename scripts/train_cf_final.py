"""
Complete Collaborative Filtering Implementation with Proper Serialization
Pure Python - No External Dependencies Needed
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Define the SimpleCF class at the module level for proper serialization
class SimpleCF:
    """
    Simple Collaborative Filtering using user-based cosine similarity
    """
    
    def __init__(self):
        self.user_ratings = defaultdict(dict)
        self.crop_ratings = defaultdict(dict)
        self.user_vectors = {}
        self.crop_index = {}
        self.user_similarities = {}
    
    def fit(self, ratings_df):
        """Train the model by building rating matrices"""
        print("🔄 Building rating matrices...")
        
        # Build rating dictionaries
        for _, row in ratings_df.iterrows():
            user_id = row['user_id']
            crop = row['crop']
            rating = row['rating']
            
            self.user_ratings[user_id][crop] = rating
            self.crop_ratings[crop][user_id] = rating
        
        print(f"✅ Built rating matrix: {len(self.user_ratings)} users, {len(self.crop_ratings)} crops")
        
        # Build user vectors for similarity calculation
        all_crops = sorted(list(set().union(*[set(ratings.keys()) for ratings in self.user_ratings.values()])))
        self.crop_index = {crop: idx for idx, crop in enumerate(all_crops)}
        
        for user_id, ratings in self.user_ratings.items():
            vector = np.zeros(len(all_crops))
            for crop, rating in ratings.items():
                vector[self.crop_index[crop]] = rating
            self.user_vectors[user_id] = vector
        
        # Compute similarities
        self._compute_similarities()
        
        return self
    
    def _compute_similarities(self):
        """Compute cosine similarities between all users"""
        print("🔄 Computing user similarities...")
        user_ids = list(self.user_vectors.keys())
        
        if len(user_ids) < 2:
            print("⚠️  Not enough users for similarity computation")
            return
        
        vectors = np.array([self.user_vectors[uid] for uid in user_ids])
        
        # Manual cosine similarity calculation
        for i, user1 in enumerate(user_ids):
            self.user_similarities[user1] = {}
            vec1 = vectors[i]
            norm1 = np.linalg.norm(vec1)
            
            for j, user2 in enumerate(user_ids):
                if i != j:
                    vec2 = vectors[j]
                    norm2 = np.linalg.norm(vec2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                        self.user_similarities[user1][user2] = similarity
                    else:
                        self.user_similarities[user1][user2] = 0
        
        print("✅ User similarities computed")
    
    def predict(self, user_id, crop, min_similarity=0.1):
        """
        Predict rating for user-crop pair
        """
        if user_id not in self.user_ratings:
            return None
            
        # Return actual rating if user already rated this crop
        if crop in self.user_ratings[user_id]:
            return self.user_ratings[user_id][crop]
        
        # Find similar users who rated this crop
        similar_users = []
        for other_user, rating in self.crop_ratings.get(crop, {}).items():
            if other_user != user_id:
                similarity = self.user_similarities.get(user_id, {}).get(other_user, 0)
                if similarity >= min_similarity:
                    similar_users.append((similarity, rating))
        
        if not similar_users:
            return None
        
        # Weighted average prediction
        total_similarity = sum(sim for sim, _ in similar_users)
        weighted_sum = sum(sim * rating for sim, rating in similar_users)
        
        predicted_rating = weighted_sum / total_similarity
        
        # Ensure reasonable bounds
        return max(1.0, min(5.0, predicted_rating))
    
    def get_user_history(self, user_id):
        """Get rating history for a user"""
        return self.user_ratings.get(user_id, {})
    
    def get_similar_users(self, user_id, top_n=3):
        """Get most similar users"""
        if user_id not in self.user_similarities:
            return []
        
        similarities = sorted(
            self.user_similarities[user_id].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return similarities
    
    def recommend_for_user(self, user_id, top_n=5):
        """Generate recommendations for a user"""
        if user_id not in self.user_ratings:
            return []
        
        user_rated = set(self.user_ratings[user_id].keys())
        all_crops = set(self.crop_ratings.keys())
        unrated_crops = all_crops - user_rated
        
        predictions = []
        for crop in unrated_crops:
            score = self.predict(user_id, crop)
            if score is not None:
                predictions.append((crop, score))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]

def train_cf_complete(min_ratings=20):
    """
    Complete training pipeline for Simple CF
    """
    print("="*60)
    print("🤖 COMPLETE COLLABORATIVE FILTERING TRAINING")
    print("="*60)
    
    # Load data
    ratings_file = 'data/crop_ratings.csv'
    
    if not os.path.exists(ratings_file):
        print(f"❌ Error: {ratings_file} not found")
        return False
    
    df_ratings = pd.read_csv(ratings_file)
    print(f"📊 Loaded {len(df_ratings)} ratings")
    
    if len(df_ratings) < min_ratings:
        print(f"⚠️  Only {len(df_ratings)} ratings (need {min_ratings})")
        return False
    
    # Display stats
    print(f"\n📈 Dataset Statistics:")
    print(f"   Users: {df_ratings['user_id'].nunique()}")
    print(f"   Crops: {df_ratings['crop'].nunique()}")
    print(f"   Rating range: {df_ratings['rating'].min():.1f} - {df_ratings['rating'].max():.1f}")
    print(f"   Average rating: {df_ratings['rating'].mean():.2f}")
    
    # Train model
    print(f"\n🔄 Training model...")
    model = SimpleCF()
    model.fit(df_ratings)
    
    # Test the model
    print(f"\n🧪 Model Testing:")
    
    # Test predictions
    test_user = df_ratings['user_id'].iloc[0]
    test_crop = df_ratings['crop'].iloc[0]
    actual_rating = df_ratings['rating'].iloc[0]
    
    prediction = model.predict(test_user, test_crop)
    print(f"   Prediction test: {test_user} + {test_crop}")
    print(f"     Actual: {actual_rating:.1f}, Predicted: {prediction:.2f}")
    
    # Test recommendations
    recommendations = model.recommend_for_user(test_user, top_n=3)
    if recommendations:
        print(f"   Recommendations for {test_user}:")
        for crop, score in recommendations:
            print(f"     🌱 {crop}: {score:.2f}")
    
    # Save model with proper serialization
    model_path = 'models/cf_model.pkl'
    os.makedirs('models', exist_ok=True)
    
    # Create a complete package with the class definition
    model_package = {
        'model_object': model,
        'model_type': 'SimpleCF',
        'metadata': {
            'n_ratings': len(df_ratings),
            'n_users': df_ratings['user_id'].nunique(),
            'n_crops': df_ratings['crop'].nunique(),
            'trained_at': pd.Timestamp.now().isoformat(),
            'rating_stats': {
                'min': df_ratings['rating'].min(),
                'max': df_ratings['rating'].max(),
                'mean': df_ratings['rating'].mean()
            }
        }
    }
    
    joblib.dump(model_package, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    print("🎉 TRAINING COMPLETE!")
   
    
    return True

if __name__ == "__main__":
    success = train_cf_complete(min_ratings=20)
    sys.exit(0 if success else 1)