"""
Generate initial synthetic user interactions for cold start
This simulates realistic farmer behavior patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import append_interaction, append_rating, initialize_csv_files

# Initialize CSVs
initialize_csv_files()

# Load training data to understand crop-soil relationships
df_train = pd.read_csv('data/processed/crop_data_cleaned.csv')

# Get list of all crops
crops = df_train['label'].unique().tolist()

print(f"🌾 Found {len(crops)} crops in dataset")
print(f"Crops: {crops}\n")

# Define user personas (realistic farmer profiles)
user_personas = {
    'FARMER_KE_001': {
        'region': 'Makueni, Kenya',
        'preferences': ['chickpea', 'pigeonpeas', 'maize'],  # Drought-tolerant
        'soil_type': 'semi_arid',
        'budget': 'low',
    },
    'FARMER_KE_002': {
        'region': 'Kiambu, Kenya',
        'preferences': ['coffee', 'banana', 'maize'],  # Highland crops
        'soil_type': 'highland',
        'budget': 'medium',
    },
    'FARMER_KE_003': {
        'region': 'Kisumu, Kenya',
        'preferences': ['rice', 'cotton', 'mango'],  # Lowland/wetland
        'soil_type': 'lowland',
        'budget': 'medium',
    },
    'FARMER_KE_004': {
        'region': 'Meru, Kenya',
        'preferences': ['coffee', 'tea', 'banana'],  # High-value crops
        'soil_type': 'highland',
        'budget': 'high',
    },
    'FARMER_KE_005': {
        'region': 'Machakos, Kenya',
        'preferences': ['lentil', 'chickpea', 'mungbean'],  # Legumes
        'soil_type': 'semi_arid',
        'budget': 'low',
    },
}

# Soil profiles for each region type
soil_profiles = {
    'semi_arid': {
        'N': (20, 40),
        'P': (30, 50),
        'K': (40, 60),
        'temperature': (26, 32),
        'humidity': (40, 55),
        'ph': (6.5, 7.5),
        'rainfall': (50, 80),
    },
    'highland': {
        'N': (40, 70),
        'P': (35, 60),
        'K': (45, 70),
        'temperature': (15, 22),
        'humidity': (60, 75),
        'ph': (5.5, 6.5),
        'rainfall': (100, 150),
    },
    'lowland': {
        'N': (50, 80),
        'P': (40, 65),
        'K': (50, 75),
        'temperature': (24, 30),
        'humidity': (70, 85),
        'ph': (6.0, 7.0),
        'rainfall': (120, 180),
    },
}

def generate_soil_params(soil_type):
    """Generate realistic soil parameters for region"""
    profile = soil_profiles[soil_type]
    
    return {
        'N': np.random.uniform(*profile['N']),
        'P': np.random.uniform(*profile['P']),
        'K': np.random.uniform(*profile['K']),
        'temperature': np.random.uniform(*profile['temperature']),
        'humidity': np.random.uniform(*profile['humidity']),
        'ph': np.random.uniform(*profile['ph']),
        'rainfall': np.random.uniform(*profile['rainfall']),
    }

def get_suitable_crops_for_params(soil_params, top_n=5):
    """Find crops that match these soil parameters"""
    from src.models.predict import CropPredictor
    
    predictor = CropPredictor()
    result = predictor.recommend_crop(**soil_params)
    
    # Get top N predictions
    all_preds = result.get('all_predictions', result['top_3_recommendations'])
    return [(crop, conf) for crop, conf in all_preds[:top_n]]

# Generate interactions for each user
print("🚀 Generating synthetic interactions...\n")

total_interactions = 0
total_ratings = 0

for user_id, persona in user_personas.items():
    print(f"👤 Generating data for {user_id} ({persona['region']})")
    
    # Generate 5-10 interactions per user
    n_interactions = np.random.randint(5, 11)
    
    for i in range(n_interactions):
        # Generate soil params for this user's region
        soil_params = generate_soil_params(persona['soil_type'])
        
        # Get suitable crops
        suitable_crops = get_suitable_crops_for_params(soil_params)
        
        if not suitable_crops:
            continue
        
        recommended_crop = suitable_crops[0][0]
        confidence = suitable_crops[0][1]
        
        # Determine action based on preferences
        if recommended_crop in persona['preferences']:
            # User likes this crop - plant it
            action = f"planted_{recommended_crop}"
            rating = np.random.uniform(4.0, 5.0)  # Positive rating
        elif np.random.random() < 0.3:
            # 30% chance user rejects
            action = "rejected"
            rating = np.random.uniform(1.0, 2.5)  # Negative rating
        else:
            # User requests alternative
            action = "requested_alternative"
            rating = np.random.uniform(2.5, 3.5)  # Neutral rating
        
        # Log interaction
        interaction_id = append_interaction(
            user_id=user_id,
            soil_params=soil_params,
            recommended_crop=recommended_crop,
            confidence=confidence,
            method='svm_only',  # Initial data is SVM-based
            action=action,
            location=persona['region']
        )
        
        # Log rating
        append_rating(
            user_id=user_id,
            crop=recommended_crop,
            rating=rating,
            rating_type='explicit' if 'planted' in action else 'implicit',
            interaction_id=interaction_id
        )
        
        total_interactions += 1
        total_ratings += 1
    
    print(f"  ✅ Generated {n_interactions} interactions")

print()
print(f"✅ Generated {total_interactions} interactions")
print(f"✅ Generated {total_ratings} ratings")
print(f"✅ Data saved to:")
print(f"   - data/user_interactions.csv")
print(f"   - data/crop_ratings.csv")
print("\n🎉 Cold start data generation complete!")
print("📌 Next step: Run 'python scripts/train_cf.py' to train CF model")