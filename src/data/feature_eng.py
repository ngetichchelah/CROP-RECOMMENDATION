"""
Feature engineering for crop recommendation
"""

import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Create derived features from raw inputs
    
    """
    df = df.copy()
    
    # 1. Nutrient Ratios
    #to show relative nutrient balance — eg, a high NPK ratio may mean nitrogen dominates the soil composition.
    df['NPK_ratio'] = df['N'] / (df['P'] + df['K'] + 1)  # Add 1 to avoid division by zero
    df['NP_ratio'] = df['N'] / (df['P'] + 1)
    df['NK_ratio'] = df['N'] / (df['K'] + 1)
    df['PK_ratio'] = df['P'] / (df['K'] + 1)
    
    # 2. Nutrient Balance Indicators
    # measure whether soil nutrients are evenly distributed or skewed. - A smaller nutrient_balance = more balanced soil.
    df['nutrient_sum'] = df['N'] + df['P'] + df['K']
    df['nutrient_balance'] = df[['N', 'P', 'K']].std(axis=1)  # Lower = more balanced
    df['nutrient_dominance'] = df[['N', 'P', 'K']].max(axis=1) / (df['nutrient_sum'] + 1)
    
    # 3. Climate Interactions
    # to simulate how temperature, humidity, and rainfall interact — helpful for modeling climate suitability.
    df['temp_humidity_index'] = df['temperature'] * df['humidity'] / 100
    df['heat_stress_index'] = df['temperature'] * (100 - df['humidity']) / 100
    df['water_stress_index'] = df['rainfall'] / (df['temperature'] ) # + 1)
    
    # 4. Growing Condition Indicators
    #to Approximate water availability and evaporation rate, critical for crop growth.
    df['moisture_availability'] = df['rainfall'] * df['humidity'] / 100
    df['evapotranspiration'] = df['temperature'] * (100 - df['humidity']) / df['rainfall'].replace(0, 1)
    
    # 5. Soil Quality Indicators
    # to Encode soil acidity and alkalinity, since crops prefer different pH ranges.
    df['ph_deviation_neutral'] = abs(df['ph'] - 7.0)  # Distance from neutral pH
    df['acidic_soil'] = (df['ph'] < 6.5).astype(int)
    df['alkaline_soil'] = (df['ph'] > 7.5).astype(int)
    
    # 6. Climate Zones (Categorical)
    #Assigns a simple climate label based on temperature thresholds.
    df['climate_zone'] = 'temperate'
    df.loc[df['temperature'] > 30, 'climate_zone'] = 'tropical'
    df.loc[df['temperature'] < 15, 'climate_zone'] = 'cool'
    #to create rainfall categories (low, moderate, high).
    df['rainfall_category'] = 'moderate'
    df.loc[df['rainfall'] < 80, 'rainfall_category'] = 'low'
    df.loc[df['rainfall'] > 200, 'rainfall_category'] = 'high'
    
    # 7. Nutrient Categories
    df['N_category'] = pd.cut(df['N'], bins=[0, 40, 80, 150], labels=['low', 'medium', 'high'])
    df['P_category'] = pd.cut(df['P'], bins=[0, 40, 80, 150], labels=['low', 'medium', 'high'])
    df['K_category'] = pd.cut(df['K'], bins=[0, 40, 80, 210], labels=['low', 'medium', 'high'])
    
    # 8. Combined Suitability Indices
    #Boolean to indicate whether conditions fit tropical, temperate, or arid regions.
    df['tropical_suitability'] = (df['temperature'] > 25).astype(int) * (df['rainfall'] > 150).astype(int) * (df['humidity'] > 70).astype(int)
    df['temperate_suitability'] = (df['temperature'].between(15, 25)).astype(int) * (df['rainfall'].between(80, 180)).astype(int)
    df['arid_suitability'] = (df['rainfall'] < 80).astype(int) * (df['temperature'] > 20).astype(int)
    
    return df

def select_best_features(df, target_col='label', n_features=15):
    """
    Select top n features based on importance
    
    Parameters:
    -----------
    df : DataFrame - Data with engineered features
    target_col : str - Target column name
    n_features : int - Number of top features to select
    
    Returns:
    --------
    list : Selected feature names
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare data
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    y = LabelEncoder().fit_transform(df[target_col])
    
    # Train RF to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top features
    top_features = importance_df.head(n_features)['feature'].tolist()
    
    print(f"Top {n_features} Features:")
    print(importance_df.head(n_features))
    
    return top_features

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/crop_data_cleaned.csv')
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Save engineered dataset
    df_engineered.to_csv('data/processed/crop_data_with_features.csv', index=False)
    
    print(f"Original features: {df.shape[1]}")
    print(f"With engineered features: {df_engineered.shape[1]}")
    print(f"New features added: {df_engineered.shape[1] - df.shape[1]}")
    
    # Select best features
    best_features = select_best_features(df_engineered, target_col='label', n_features=15)