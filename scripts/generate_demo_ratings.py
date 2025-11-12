"""
Generate synthetic ratings for bootstrapping CF model
Creates ~200 plausible user-crop-rating combinations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.helpers import initialize_csv_files

# Regions
# Regions and their typical successful crops
REGIONS = {
    'Makueni': ['chickpea', 'mothbeans', 'pigeonpeas', 'groundnuts', 'sorghum'],
    'Kitui': ['sorghum', 'millet', 'cowpeas', 'pigeonpeas', 'cassava'],
    'Machakos': ['maize', 'beans', 'pigeonpeas', 'greengram', 'sorghum'],
    'Embu': ['rice', 'maize', 'beans', 'coffee', 'tea'],
    'Meru': ['coffee', 'tea', 'maize', 'beans', 'banana']
}

# All crops in system
ALL_CROPS = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
    'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
    'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
    'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 
    'coffee', 'groundnuts'
]

def generate_user_id(region, farmer_num):
    """Generate realistic user ID"""
    region_code = region[:3].upper()
    return f"{region_code}_F{farmer_num:03d}"

def generate_ratings(n_users_per_region=8, crops_per_user=5):
    """
    Generate synthetic ratings
    
    Parameters:
    -----------
    n_users_per_region : int
        Number of farmers per region
    crops_per_user : int
        Average crops rated per farmer
    
    Returns:
    --------
    DataFrame with ratings
    """
    ratings = []
    
    for region, suitable_crops in REGIONS.items():
        for farmer_num in range(1, n_users_per_region + 1):
            user_id = generate_user_id(region, farmer_num)
            
            # How many crops has this farmer tried?
            n_crops = np.random.randint(3, crops_per_user + 3)
            
            # Select crops (80% from suitable, 20% random experiments)
            n_suitable = int(n_crops * 0.8)
            n_random = n_crops - n_suitable
            
            selected_crops = []
            
            # Add suitable crops with high ratings
            if suitable_crops:
                selected_suitable = np.random.choice(
                    suitable_crops, 
                    size=min(n_suitable, len(suitable_crops)),
                    replace=False
                ).tolist()
                selected_crops.extend(selected_suitable)
            
            # Add random crops with lower ratings
            other_crops = [c for c in ALL_CROPS if c not in suitable_crops]
            if other_crops and n_random > 0:
                selected_random = np.random.choice(
                    other_crops,
                    size=min(n_random, len(other_crops)),
                    replace=False
                ).tolist()
                selected_crops.extend(selected_random)
            
            # Generate ratings for each crop
            for crop in selected_crops:
                # Suitable crops get higher ratings (3.5-5.0)
                # Random experiments get lower ratings (1.0-3.5)
                if crop in suitable_crops:
                    rating = np.random.uniform(3.5, 5.0)
                else:
                    rating = np.random.uniform(1.5, 3.5)
                
                # Round to 0.5 increments
                rating = round(rating * 2) / 2
                
                # Random timestamp within last 12 months
                days_ago = np.random.randint(1, 365)
                timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()
                
                ratings.append({
                    'user_id': user_id,
                    'crop': crop,
                    'rating': rating,
                    'rating_type': 'synthetic',
                    'interaction_id': '',
                    'timestamp': timestamp
                })
    
    return pd.DataFrame(ratings)

def main():
    """Generate and save demo ratings"""
    
    print("GENERATING DEMO RATINGS FOR CF BOOTSTRAPPING")
        
    # Ensure CSV files exist
    initialize_csv_files()
    
    # Generate ratings
    print("\nGenerating synthetic ratings...")
    ratings_df = generate_ratings(
        n_users_per_region=8,  # 8 farmers per region
        crops_per_user=5       # ~5 crops per farmer
    )
    
    print(f"✅ Generated {len(ratings_df)} ratings")
    print(f"   - Unique users: {ratings_df['user_id'].nunique()}")
    print(f"   - Unique crops: {ratings_df['crop'].nunique()}")
    print(f"   - Rating distribution:")
    print(ratings_df['rating'].value_counts().sort_index().to_string())
    
    # Check if ratings.csv already has data
    existing_ratings = pd.read_csv('data/ratings.csv')
    
    if len(existing_ratings) > 0:
        response = input(f"\n⚠️  ratings.csv already has {len(existing_ratings)} entries. Append? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Save to CSV
    ratings_df.to_csv('data/ratings.csv', mode='a', header=False, index=False)
    
    print(f"\n✅ Saved to data/ratings.csv")
    print(f"\nTotal ratings now: {len(existing_ratings) + len(ratings_df)}")
    
    print()
    print("NEXT STEPS")
    print()
    print("1. Train CF model:")
    print("   python scripts/retrain_cf.py")
    print("\n2. Restart Streamlit app:")
    print("   streamlit run app/streamlit_app.py")
    print("\n3. Hybrid recommendations will now be active!")

if __name__ == "__main__":
    main()