"""
Fix crop_requirements_summary.csv to have correct format for clustering
"""

import pandas as pd

print()
print("FIXING CROP REQUIREMENTS CSV")
print()

# Load the cleaned data
df = pd.read_csv('data/processed/crop_data_cleaned.csv')

print(f"Loaded data: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check if 'label' or 'crop' column exists
if 'label' in df.columns:
    target_col = 'label'
elif 'crop' in df.columns:
    target_col = 'crop'
else:
    print("Error: No 'label' or 'crop' column found!")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

print(f"\nUsing '{target_col}' as crop name column")

# Create crop requirements summary with CORRECT structure
crop_summary = df.groupby(target_col).agg({
    'N': ['mean', 'min', 'max', 'std'],
    'P': ['mean', 'min', 'max', 'std'],
    'K': ['mean', 'min', 'max', 'std'],
    'temperature': ['mean', 'min', 'max', 'std'],
    'humidity': ['mean', 'min', 'max', 'std'],
    'ph': ['mean', 'min', 'max', 'std'],
    'rainfall': ['mean', 'min', 'max', 'std']
}).reset_index()

# Flatten column names
crop_summary.columns = [
    'label',  # Always use 'label' for consistency
    'N_avg', 'N_min', 'N_max', 'N_std',
    'P_avg', 'P_min', 'P_max', 'P_std',
    'K_avg', 'K_min', 'K_max', 'K_std',
    'temp_avg', 'temp_min', 'temp_max', 'temp_std',
    'humidity_avg', 'humidity_min', 'humidity_max', 'humidity_std',
    'ph_avg', 'ph_min', 'ph_max', 'ph_std',
    'rainfall_avg', 'rainfall_min', 'rainfall_max', 'rainfall_std'
]

# Save with correct format
crop_summary.to_csv('data/processed/crop_requirements_summary.csv', index=False)

print()
print(" FIXED crop_requirements_summary.csv")
print()
print(f"Shape: {crop_summary.shape}")
print(f"Columns ({len(crop_summary.columns)}): {list(crop_summary.columns)}")
print(f"\nFirst 3 rows:")
print(crop_summary.head(3))
print()

# Verify the file
print("\n Verifying file can be read correctly...")
test_df = pd.read_csv('data/processed/crop_requirements_summary.csv')
print(f"Verification passed! Shape: {test_df.shape}")
print(f" Columns: {list(test_df.columns)}")
