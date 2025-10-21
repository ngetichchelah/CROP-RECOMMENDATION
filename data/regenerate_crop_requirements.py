"""
Regenerate crop_requirements_summary.csv with correct columns
"""

import pandas as pd
import numpy as np

print("Regenerating crop_requirements_summary.csv...")
print("=" * 60)

# Load the cleaned data
df = pd.read_csv('data/processed/crop_data_cleaned.csv')

print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# Create crop requirements summary with ALL statistics
crop_summary = df.groupby('label').agg({
    'N': ['mean', 'min', 'max', 'std'],
    'P': ['mean', 'min', 'max', 'std'],
    'K': ['mean', 'min', 'max', 'std'],
    'temperature': ['mean', 'min', 'max', 'std'],
    'humidity': ['mean', 'min', 'max', 'std'],
    'ph': ['mean', 'min', 'max', 'std'],
    'rainfall': ['mean', 'min', 'max', 'std']
}).reset_index()

# Flatten the multi-level column names
crop_summary.columns = [
    'label',
    'N_avg', 'N_min', 'N_max', 'N_std',
    'P_avg', 'P_min', 'P_max', 'P_std',
    'K_avg', 'K_min', 'K_max', 'K_std',
    'temp_avg', 'temp_min', 'temp_max', 'temp_std',
    'humidity_avg', 'humidity_min', 'humidity_max', 'humidity_std',
    'ph_avg', 'ph_min', 'ph_max', 'ph_std',
    'rainfall_avg', 'rainfall_min', 'rainfall_max', 'rainfall_std'
]

# Save to CSV
crop_summary.to_csv('data/processed/crop_requirements_summary.csv', index=False)

print("\n✅ File regenerated successfully!")
print(f"✅ Shape: {crop_summary.shape}")
print(f"✅ Columns ({len(crop_summary.columns)}):")
for i, col in enumerate(crop_summary.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n✅ Sample data (first 3 crops):")
print(crop_summary.head(3))

print("\n✅ Saved to: data/processed/crop_requirements_summary.csv")
print("=" * 60)