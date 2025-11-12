"""
Helper functions for interaction and rating logging
"""

import pandas as pd
import os
from datetime import datetime
import uuid

# Define CSV paths
INTERACTIONS_FILE = 'data/interactions.csv'
RATINGS_FILE = 'data/ratings.csv'
MODEL_LOG_FILE = 'data/model_training_log.csv'

# CSV Headers
INTERACTIONS_HEADER = [
    'interaction_id', 'user_id', 'timestamp', 'N', 'P', 'K', 
    'temperature', 'humidity', 'ph', 'rainfall', 'recommended_crop', 
    'confidence', 'method', 'action', 'crop_planted', 'location'
]

RATINGS_HEADER = [
    'user_id', 'crop', 'rating', 'rating_type', 'interaction_id', 'timestamp'
]

MODEL_LOG_HEADER = [
    'timestamp', 'model_type', 'algorithm', 'n_ratings', 'rmse', 'filepath'
]


def initialize_csv_files():
    """
    Create CSV files with headers if they don't exist
    Safe to call multiple times
    """
    os.makedirs('data', exist_ok=True)
    
    # Initialize interactions.csv
    if not os.path.exists(INTERACTIONS_FILE):
        df = pd.DataFrame(columns=INTERACTIONS_HEADER)
        df.to_csv(INTERACTIONS_FILE, index=False)
        print(f"✅ Created {INTERACTIONS_FILE}")
    
    # Initialize ratings.csv
    if not os.path.exists(RATINGS_FILE):
        df = pd.DataFrame(columns=RATINGS_HEADER)
        df.to_csv(RATINGS_FILE, index=False)
        print(f"✅ Created {RATINGS_FILE}")
    
    # Initialize model_training_log.csv
    if not os.path.exists(MODEL_LOG_FILE):
        df = pd.DataFrame(columns=MODEL_LOG_HEADER)
        df.to_csv(MODEL_LOG_FILE, index=False)
        print(f"✅ Created {MODEL_LOG_FILE}")


def append_interaction(user_id, soil_params, recommended_crop, confidence, 
                      method, action, crop_planted=None, location=None):
    """
    Append interaction to interactions.csv
    
    Parameters:
    -----------
    user_id : str
    soil_params : dict with keys N, P, K, temperature, humidity, ph, rainfall
    recommended_crop : str
    confidence : float (0-100)
    method : str ('hybrid' or 'content_based')
    action : str ('planted_X', 'rejected', 'requested_alternative')
    crop_planted : str, optional (extracted from action if None)
    location : str, optional
    
    Returns:
    --------
    interaction_id : str
    """
    initialize_csv_files()  # Ensure file exists
    
    interaction_id = f"INT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
    timestamp = datetime.now().isoformat()
    
    # Extract crop from action if not provided
    if crop_planted is None and action.startswith('planted_'):
        crop_planted = action.replace('planted_', '')
    
    row = {
        'interaction_id': interaction_id,
        'user_id': user_id,
        'timestamp': timestamp,
        'N': soil_params['N'],
        'P': soil_params['P'],
        'K': soil_params['K'],
        'temperature': soil_params['temperature'],
        'humidity': soil_params['humidity'],
        'ph': soil_params['ph'],
        'rainfall': soil_params['rainfall'],
        'recommended_crop': recommended_crop,
        'confidence': confidence,
        'method': method,
        'action': action,
        'crop_planted': crop_planted,
        'location': location if location else ''
    }
    
    df = pd.DataFrame([row])
    df.to_csv(INTERACTIONS_FILE, mode='a', header=False, index=False)
    
    print(f"✅ Interaction {interaction_id} recorded")
    return interaction_id


def append_rating(user_id, crop, rating, rating_type='explicit', interaction_id=None):
    """
    Append rating to ratings.csv
    
    Parameters:
    -----------
    user_id : str
    crop : str
    rating : float (1-5)
    rating_type : str ('explicit' or 'implicit')
    interaction_id : str, optional
    
    Returns:
    --------
    None
    """
    initialize_csv_files()  # Ensure file exists
    
    timestamp = datetime.now().isoformat()
    
    row = {
        'user_id': user_id,
        'crop': crop,
        'rating': rating,
        'rating_type': rating_type,
        'interaction_id': interaction_id if interaction_id else '',
        'timestamp': timestamp
    }
    
    df = pd.DataFrame([row])
    df.to_csv(RATINGS_FILE, mode='a', header=False, index=False)
    
    print(f"✅ Rating recorded: {user_id} → {crop} = {rating}/5.0")


def append_model_log(model_type, algorithm, n_ratings, rmse, filepath):
    """
    Append model training log to model_training_log.csv
    
    Parameters:
    -----------
    model_type : str (e.g., 'collaborative_filtering')
    algorithm : str (e.g., 'SVD')
    n_ratings : int
    rmse : float
    filepath : str
    """
    initialize_csv_files()  # Ensure file exists
    
    timestamp = datetime.now().isoformat()
    
    row = {
        'timestamp': timestamp,
        'model_type': model_type,
        'algorithm': algorithm,
        'n_ratings': n_ratings,
        'rmse': rmse,
        'filepath': filepath
    }
    
    df = pd.DataFrame([row])
    df.to_csv(MODEL_LOG_FILE, mode='a', header=False, index=False)
    
    print(f"✅ Model log recorded: {algorithm} RMSE={rmse:.4f}")


def get_user_history(user_id):
    """
    Get user's interaction history
    
    Returns:
    --------
    list of dicts or None if no history
    """
    if not os.path.exists(INTERACTIONS_FILE):
        return None
    
    try:
        df = pd.read_csv(INTERACTIONS_FILE)
        user_data = df[df['user_id'] == user_id]
        
        if user_data.empty:
            return None
        
        return user_data.to_dict('records')
    except Exception as e:
        print(f"⚠️ Error reading user history: {e}")
        return None


def get_ratings_count():
    """Get total number of ratings in database"""
    if not os.path.exists(RATINGS_FILE):
        return 0
    
    try:
        df = pd.read_csv(RATINGS_FILE)
        return len(df)
    except:
        return 0