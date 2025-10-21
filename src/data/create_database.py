"""
Create SQLite database for crop recommendation system
"""

import sqlite3
#built-in Python library for working with SQLite databases.
import pandas as pd
import os

def create_database():
    """Create SQLite database and load data"""
    
    # Paths
    db_path = 'data/database/crop_recommendation.db'
    data_path = 'data/processed/crop_data_cleaned.csv'
    
    # Ensure database directory exists
    os.makedirs('data/database', exist_ok=True)
    
    # Load data
    print("Loading cleaned data...")
    df = pd.read_csv(data_path)
    
    # Create database connection
    print(f"Creating database at: {db_path}")
    conn = sqlite3.connect(db_path) #used to execute SQL commands.
    
    # Create main crops table
    print("Creating 'crops' table...")
    df.to_sql('crops', conn, if_exists='replace', index=False)
    
    # Create crop requirements summary table
    print("Creating 'crop_requirements' table...")
    crop_summary = df.groupby('label').agg({
        'N': ['mean', 'min', 'max', 'std'],
        'P': ['mean', 'min', 'max', 'std'],
        'K': ['mean', 'min', 'max', 'std'],
        'temperature': ['mean', 'min', 'max', 'std'],
        'humidity': ['mean', 'min', 'max', 'std'],
        'ph': ['mean', 'min', 'max', 'std'],
        'rainfall': ['mean', 'min', 'max', 'std']
    }).reset_index()
    
    #agg() function creates multi-level column names like ('N', 'mean').
    # so there is need to flatten column names
    crop_summary.columns = ['crop',
                            'N_avg', 'N_min', 'N_max', 'N_std',
                            'P_avg', 'P_min', 'P_max', 'P_std',
                            'K_avg', 'K_min', 'K_max', 'K_std',
                            'temp_avg', 'temp_min', 'temp_max', 'temp_std',
                            'humidity_avg', 'humidity_min', 'humidity_max', 'humidity_std',
                            'ph_avg', 'ph_min', 'ph_max', 'ph_std',
                            'rainfall_avg', 'rainfall_min', 'rainfall_max', 'rainfall_std']
    
    crop_summary.to_sql('crop_requirements', conn, if_exists='replace', index=False)
    
    # Create indexes for better query performance
    #CREATE INDEX ensures efficient lookups for frequent query fields.
    print("Creating indexes...")
    cursor = conn.cursor()
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_crop_label ON crops(label)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_nitrogen ON crops(N)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_temperature ON crops(temperature)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rainfall ON crops(rainfall)')
    
    conn.commit()
    
    # Verify tables
    print("DATABASE CREATED SUCCESSFULLY!")
  
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'") #Lists all tables in the database.
    tables = cursor.fetchall()
    print(f"\nTables created: {[table[0] for table in tables]}")
    
    # Show row counts
    #Counts the number of records in each table to confirm data loaded correctly.
    cursor.execute("SELECT COUNT(*) FROM crops")
    crops_count = cursor.fetchone()[0]
    print(f"  - crops table: {crops_count} rows")
    
    cursor.execute("SELECT COUNT(*) FROM crop_requirements")
    req_count = cursor.fetchone()[0]
    print(f"  - crop_requirements table: {req_count} rows")
    
    
    conn.close()
    return db_path

if __name__ == "__main__":
    db_path = create_database()
    print(f"\n Database location: {db_path}")