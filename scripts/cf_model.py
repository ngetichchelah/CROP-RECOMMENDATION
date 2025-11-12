"""
Standalone Collaborative Filtering Model
No serialization issues - can be imported anywhere
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import joblib
import os

class SimpleCF:
    """
    Simple Collaborative Filtering using user-based cosine similarity
    Standalone class that can be properly serialized
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

def train_cf_model():
    """Train and save the CF model"""
    #from cf_model import SimpleCF  # Import from this module
    
    print("🤖 Training Collaborative Filtering Model...")
    
    # Load ratings
    ratings_file = 'data/crop_ratings.csv'
    if not os.path.exists(ratings_file):
        print(f"❌ {ratings_file} not found")
        return False
    
    df_ratings = pd.read_csv(ratings_file)
    print(f"📊 Loaded {len(df_ratings)} ratings")
    
    if len(df_ratings) < 20:
        print(f"⚠️  Only {len(df_ratings)} ratings (need at least 20)")
        return False
    
    # Train model
    model = SimpleCF()
    model.fit(df_ratings)
    
    # Save model
    model_path = 'models/cf_model.pkl'
    os.makedirs('models', exist_ok=True)
    
    model_package = {
        'model_object': model,
        'model_type': 'SimpleCF',
        'metadata': {
            'n_ratings': len(df_ratings),
            'n_users': df_ratings['user_id'].nunique(),
            'n_crops': df_ratings['crop'].nunique(),
            'trained_at': pd.Timestamp.now().isoformat()
        }
    }
    
    joblib.dump(model_package, model_path)
    print(f"💾 Model saved to: {model_path}")
    return True

if __name__ == "__main__":
    train_cf_model()