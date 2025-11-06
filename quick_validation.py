"""
Quick validation: Does Makueni fall within training data range?
"""

import pandas as pd
import numpy as np

print()
print("QUICK VALIDATION: GENERALIZATION CHECK")
print()

# Load training data
df = pd.read_csv('data/processed/crop_data_cleaned.csv')

# Define Makueni typical conditions (from local agricultural reports)
makueni_conditions = {
    'N': 35,  # kg/ha (low, semi-arid)
    'P': 45,
    'K': 25,
    'temperature': 28,  # °C (hot)
    'humidity': 45,  # % (low)
    'ph': 6.8,  # neutral
    'rainfall': 60  # mm (low - drought-prone)
}

print("\n📍 Makueni Typical Conditions:")
for param, value in makueni_conditions.items():
    print(f"  {param}: {value}")

print("\n📊 Training Data Range vs Makueni:")
print()
results = []

for param, makueni_val in makueni_conditions.items():
    train_min = df[param].min()
    train_max = df[param].max()
    train_mean = df[param].mean()
    train_std = df[param].std()
    
    # Check if Makueni is within range
    within_range = train_min <= makueni_val <= train_max
    
    # Check if it's typical (within 2 std of mean)
    z_score = abs(makueni_val - train_mean) / train_std
    typical = z_score <= 2
    
    status = "✅" if within_range and typical else "⚠️" if within_range else "❌"
    
    print(f"{param:15} | Train: [{train_min:6.1f}, {train_max:6.1f}] | "
          f"Makueni: {makueni_val:6.1f} | {status}")
    
    results.append({
        'parameter': param,
        'within_range': within_range,
        'typical': typical,
        'z_score': z_score
    })

# Summary
results_df = pd.DataFrame(results)
in_range = results_df['within_range'].sum()
typical_count = results_df['typical'].sum()

print()
print("VALIDATION SUMMARY")
print()
print(f"Parameters within training range: {in_range}/7 ({in_range/7*100:.0f}%)")
print(f"Parameters typical (±2σ): {typical_count}/7 ({typical_count/7*100:.0f}%)")

if in_range == 7 and typical_count >= 5:
    print("\n✅ PASS - Makueni conditions well-represented in training data")
    print("   Model should generalize well")
elif in_range == 7:
    print("\n⚠️ CAUTION - All parameters in range but some unusual")
    print("   Model may work but recommend validation")
else:
    print("\n❌ FAIL - Some parameters outside training range")
    print("   Model NOT validated for this region")
    print("   MUST calibrate before deployment")

# Identify most concerning parameters
concerning = results_df[results_df['z_score'] > 2]
if len(concerning) > 0:
    print("\n⚠️ Parameters of concern:")
    for _, row in concerning.iterrows():
        print(f"  - {row['parameter']}: z-score = {row['z_score']:.2f}")

print()

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

for i, param in enumerate(features):
    ax = axes[i]
    
    # Training data distribution
    ax.hist(df[param], bins=30, alpha=0.6, color='steelblue', 
            label='Training data', edgecolor='black')
    
    # Makueni value
    makueni_val = makueni_conditions[param]
    ax.axvline(makueni_val, color='red', linewidth=3, 
               linestyle='--', label='Makueni', alpha=0.8)
    
    # Mean ± 2σ range
    mean = df[param].mean()
    std = df[param].std()
    ax.axvspan(mean - 2*std, mean + 2*std, alpha=0.2, 
               color='green', label='Typical range (±2σ)')
    
    ax.set_title(f'{param.upper()}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Remove empty subplot
axes[-1].remove()

plt.suptitle('Training Data vs Makueni Conditions', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/figures/generalization_check.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: results/figures/generalization_check.png")
plt.show()