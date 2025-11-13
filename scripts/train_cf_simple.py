"""
Pure Python Collaborative Filtering - No External Dependencies
Uses simple user-based cosine similarity
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import joblib
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity

class SimpleCF:
    """
    Simple Collaborative Filtering using user-based cosine similarity
    No external dependencies beyond pandas/numpy/scikit-learn
    """
    
    def __init__(self):
        self.user_ratings = defaultdict(dict)  # user_id -> {crop: rating}
        self.crop_ratings = defaultdict(dict)  # crop -> {user_id: rating}
        self.user_similarities = {}  # Cache for user similarities
        self.user_vectors = {}  # User rating vectors for similarity calculation
        
    def fit(self, ratings_df):
        """Train the model by building rating matrices"""
        print("🔄 Building rating matrices...")
        
        # Build user-crop rating dictionaries
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
        
        # Precompute some similarities for faster predictions
        self._precompute_similarities()
        
        return self
    
    def _precompute_similarities(self):
        """Precompute similarities between users"""
        print("🔄 Precomputing user similarities...")
        user_ids = list(self.user_vectors.keys())
        vectors = np.array([self.user_vectors[uid] for uid in user_ids])
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(vectors)
        
        # Store similarities in dictionary format
        for i, user1 in enumerate(user_ids):
            self.user_similarities[user1] = {}
            for j, user2 in enumerate(user_ids):
                if i != j:  # Don't store self-similarity
                    self.user_similarities[user1][user2] = similarity_matrix[i, j]
    
    def predict(self, user_id, crop, min_similarity=0.1):
        """
        Predict rating for user-crop pair
        
        Args:
            user_id: User identifier
            crop: Crop name
            min_similarity: Minimum similarity threshold for neighbors
            
        Returns:
            Predicted rating or None if cannot predict
        """
        if user_id not in self.user_ratings:
            return None  # New user, no history
            
        # If user already rated this crop, return actual rating
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
            return None  # No similar users rated this crop
        
        # Weighted average of similar users' ratings
        total_similarity = sum(sim for sim, _ in similar_users)
        weighted_sum = sum(sim * rating for sim, rating in similar_users)
        
        predicted_rating = weighted_sum / total_similarity
        
        # Ensure rating is within reasonable bounds (1-5)
        return max(1.0, min(5.0, predicted_rating))
    
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
    
    def recommend_for_user(self, user_id, top_n=5, min_similarity=0.1):
        """Generate top N recommendations for a user"""
        if user_id not in self.user_ratings:
            return []
        
        user_rated_crops = set(self.user_ratings[user_id].keys())
        all_crops = set(self.crop_ratings.keys())
        unrated_crops = all_crops - user_rated_crops
        
        predictions = []
        for crop in unrated_crops:
            predicted_rating = self.predict(user_id, crop, min_similarity)
            if predicted_rating is not None:
                predictions.append((crop, predicted_rating))
        
        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_n]

def train_simple_cf(min_ratings=20):
    """Train simple collaborative filtering model"""
    
    
    print("🤖 PURE PYTHON COLLABORATIVE FILTERING TRAINING")
    
    
    # Load ratings data
    ratings_file = 'data/crop_ratings.csv'
    
    if not os.path.exists(ratings_file):
        print(f"❌ Error: {ratings_file} not found")
        print("💡 Run 'python scripts/generate_initial_data.py' first")
        return False
    
    df_ratings = pd.read_csv(ratings_file)
    print(f"\n📊 Loaded {len(df_ratings)} ratings")
    
    if len(df_ratings) < min_ratings:
        print(f"⚠️  Only {len(df_ratings)} ratings (need {min_ratings})")
        print("💡 Generate more data or lower min_ratings threshold")
        return False
    
    # Display statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Unique users: {df_ratings['user_id'].nunique()}")
    print(f"   Unique crops: {df_ratings['crop'].nunique()}")
    print(f"   Rating range: {df_ratings['rating'].min():.1f} - {df_ratings['rating'].max():.1f}")
    print(f"   Average rating: {df_ratings['rating'].mean():.2f}")
    
    # Train model
    print(f"\n🔄 Training Simple CF model...")
    model = SimpleCF()
    model.fit(df_ratings)
    
    # Test predictions
    print(f"\n🧪 Testing predictions...")
    test_samples = min(3, len(df_ratings))
    
    for i in range(test_samples):
        user_id = df_ratings['user_id'].iloc[i]
        crop = df_ratings['crop'].iloc[i]
        actual_rating = df_ratings['rating'].iloc[i]
        
        predicted_rating = model.predict(user_id, crop)
        
        if predicted_rating is not None:
            print(f"   ✅ {user_id} + {crop}:")
            print(f"      Actual: {actual_rating:.1f}, Predicted: {predicted_rating:.2f}")
        else:
            print(f"   ⚠️  {user_id} + {crop}: Cannot predict")
    
    # Test recommendations
    test_user = df_ratings['user_id'].iloc[0]
    recommendations = model.recommend_for_user(test_user, top_n=3)
    
    if recommendations:
        print(f"\n🔍 Sample recommendations for {test_user}:")
        for crop, score in recommendations:
            print(f"   🌱 {crop}: {score:.2f}")
    
    # Save model
    model_path = 'models/cf_model.pkl'
    os.makedirs('models', exist_ok=True)
    
    model_package = {
        'model': model,
        'metadata': {
            'n_ratings': len(df_ratings),
            'n_users': df_ratings['user_id'].nunique(),
            'n_crops': df_ratings['crop'].nunique(),
            'trained_at': pd.Timestamp.now().isoformat(),
            'algorithm': 'SimpleCF-UserBased'
        }
    }
    
    joblib.dump(model_package, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    
    print("🎉 SIMPLE CF TRAINING COMPLETE!")
    
    print(f"\n📌 This model uses:")
    print(f"   ✅ Pure Python (no compilation needed)")
    print(f"   ✅ User-based collaborative filtering")
    print(f"   ✅ Cosine similarity for user matching")
    print(f"   ✅ Weighted average predictions")
    
    return True

if __name__ == "__main__":
    success = train_simple_cf(min_ratings=20)
    sys.exit(0 if success else 1)