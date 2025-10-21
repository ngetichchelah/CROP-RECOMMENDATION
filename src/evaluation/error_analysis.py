"""
Error Analysis for Crop Recommendation System
Analyze model predictions to identify failure patterns
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ErrorAnalyzer:
    """Comprehensive error analysis for crop recommendation model"""
    
    def __init__(self):
        """Initialize analyzer and load model"""
        print()
        print("INITIALIZING ERROR ANALYZER")
        print()
        
        # Load model and preprocessors
        try:
            self.model = joblib.load('models/crop_model_svm.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.encoder = joblib.load('models/label_encoder.pkl')
            print("✅ Model and preprocessors loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Load data
        try:
            self.df = pd.read_csv('data/processed/crop_data_cleaned.csv')
            print(f"✅ Data loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
        
        self.predictions_df = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
    
    def generate_predictions(self, test_size=0.2, random_state=42):
        """
        Generate predictions on test set
        
        Parameters:
        -----------
        test_size : float - Proportion of data for testing
        random_state : int - Random seed for reproducibility
        
        Returns:
        --------
        DataFrame : Test data with predictions and confidence scores
        """
        print()
        print("STEP 1: GENERATING PREDICTIONS ON TEST SET")
        print()
        
        from sklearn.model_selection import train_test_split
        
        # Prepare features and target
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = self.df[features]
        y = self.df['label']
        
        # Split data (same way as training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTest set size: {len(X_test)} samples")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_test_encoded = self.encoder.transform(y_test)
        
        # Get predictions
        y_pred_encoded = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Decode predictions
        y_pred = self.encoder.inverse_transform(y_pred_encoded)
        
        # Store for later use
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        # Create comprehensive predictions dataframe
        predictions_df = X_test.copy()
        predictions_df['true_crop'] = y_test.values
        predictions_df['predicted_crop'] = y_pred
        predictions_df['is_correct'] = (y_test.values == y_pred).astype(int)
        
        # Add confidence scores
        max_proba = y_pred_proba.max(axis=1)
        predictions_df['confidence'] = max_proba * 100
        
        # Add confidence categories
        predictions_df['confidence_category'] = pd.cut(
            predictions_df['confidence'],
            bins=[0, 50, 70, 85, 95, 100],
            labels=['Very Low (<50%)', 'Low (50-70%)', 'Medium (70-85%)', 
                    'High (85-95%)', 'Very High (>95%)']
        )
        
        # Add true class probabilities
        for i, crop_name in enumerate(self.encoder.classes_):
            predictions_df[f'prob_{crop_name}'] = y_pred_proba[:, i]
        
        self.predictions_df = predictions_df
        
        # Print summary
        accuracy = (predictions_df['is_correct'].sum() / len(predictions_df)) * 100
        print(f"\n✅ Predictions generated successfully")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Correct predictions: {predictions_df['is_correct'].sum()}")
        print(f"   Incorrect predictions: {len(predictions_df) - predictions_df['is_correct'].sum()}")
        
        # Save to CSV
        predictions_df.to_csv('results/predictions_with_errors.csv', index=False)
        print(f"\n✅ Predictions saved to: results/predictions_with_errors.csv")
        
        return predictions_df
    
    def analyze_errors_by_confidence(self):
        """Analyze error rates by confidence level"""
        print()
        print("ERROR ANALYSIS BY CONFIDENCE LEVEL")
        print()
        
        if self.predictions_df is None:
            print("❌ No predictions available. Run generate_predictions() first.")
            return
        
        # Group by confidence category
        confidence_analysis = self.predictions_df.groupby('confidence_category').agg({
            'is_correct': ['count', 'sum', 'mean']
        }).round(4)
        
        confidence_analysis.columns = ['Total_Predictions', 'Correct_Predictions', 'Accuracy']
        confidence_analysis['Error_Rate'] = 1 - confidence_analysis['Accuracy']
        confidence_analysis['Accuracy'] = confidence_analysis['Accuracy'] * 100
        confidence_analysis['Error_Rate'] = confidence_analysis['Error_Rate'] * 100
        
        print("\nError Rate by Confidence Level:")
        print(confidence_analysis)
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy by confidence
        confidence_analysis['Accuracy'].plot(
            kind='bar', ax=axes[0], color='steelblue', edgecolor='black'
        )
        axes[0].set_title('Accuracy by Confidence Level', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Confidence Category', fontsize=12)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_ylim(0, 100)
        axes[0].axhline(y=93.24, color='red', linestyle='--', label='Overall Accuracy')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Error rate by confidence
        confidence_analysis['Error_Rate'].plot(
            kind='bar', ax=axes[1], color='salmon', edgecolor='black'
        )
        axes[1].set_title('Error Rate by Confidence Level', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Confidence Category', fontsize=12)
        axes[1].set_ylabel('Error Rate (%)', fontsize=12)
        axes[1].set_ylim(0, 50)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/error_by_confidence.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Figure saved: results/figures/error_by_confidence.png")
        plt.show()
        
        return confidence_analysis
    
    def get_error_summary(self):
        """Get quick summary of errors"""
        print()
        print("ERROR SUMMARY")
        print()
        
        if self.predictions_df is None:
            print("❌ No predictions available. Run generate_predictions() first.")
            return
        
        errors = self.predictions_df[self.predictions_df['is_correct'] == 0]
        
        print(f"\nTotal errors: {len(errors)}")
        print(f"Error rate: {(len(errors) / len(self.predictions_df)) * 100:.2f}%")
        
        print("\nMost commonly misclassified crops (true labels):")
        error_crops = errors['true_crop'].value_counts().head(10)
        for crop, count in error_crops.items():
            print(f"  {crop}: {count} errors")
        
        print("\nMost common incorrect predictions:")
        wrong_pred = errors['predicted_crop'].value_counts().head(10)
        for crop, count in wrong_pred.items():
            print(f"  {crop}: {count} times")
        
        print("\nAverage confidence on errors:")
        print(f"  {errors['confidence'].mean():.2f}%")
        
        print("\nAverage confidence on correct predictions:")
        correct = self.predictions_df[self.predictions_df['is_correct'] == 1]
        print(f"  {correct['confidence'].mean():.2f}%")
        
        return errors
    
    def display_sample_errors(self, n=10):
        """Display sample misclassifications for inspection"""
        print()
        print(f"SAMPLE ERRORS (First {n})")
        print()
        
        if self.predictions_df is None:
            print("❌ No predictions available. Run generate_predictions() first.")
            return
        
        errors = self.predictions_df[self.predictions_df['is_correct'] == 0]
        
        if len(errors) == 0:
            print("✅ No errors found! Perfect predictions!")
            return
        
        sample_errors = errors.head(n)
        
        for idx, row in sample_errors.iterrows():
            print(f"\nError #{idx + 1}:")
            print(f"  Input parameters:")
            print(f"    N={row['N']}, P={row['P']}, K={row['K']}")
            print(f"    Temp={row['temperature']}°C, Humidity={row['humidity']}%")
            print(f"    pH={row['ph']}, Rainfall={row['rainfall']}mm")
            print(f"  True crop: {row['true_crop']}")
            print(f"  Predicted: {row['predicted_crop']}")
            print(f"  Confidence: {row['confidence']:.2f}%")
            key = f"prob_{row['true_crop']}"
            print(f"  True crop probability: {row[key]*100:.2f}%")
            #print(f"  True crop probability: {row[f'prob_{row[\"true_crop\"]}']*100:.2f}% ")
            print("-" * 60)
        
        return sample_errors

#------------------------confusion trix -------------------------------
    def analyze_confusion_matrix(self):
        """
    Generate and analyze confusion matrix
    Shows which crops are confused with each other
    """
        print()
        print("STEP 2: CONFUSION MATRIX ANALYSIS")
        print()
    
        if self.y_test is None or self.y_pred is None:
            print("❌ No predictions available. Run generate_predictions() first.")
            return
    
    # Get unique crop names
        crops = sorted(self.encoder.classes_)
    
    # Generate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred, labels=crops)
    
    # Convert to DataFrame for easier analysis
        cm_df = pd.DataFrame(cm, index=crops, columns=crops)
        
        print(f"\nConfusion Matrix Shape: {cm.shape}")
        print(f"Crops analyzed: {len(crops)}")
        
        # Find most confused pairs
        print()
        print("TOP 10 MOST CONFUSED CROP PAIRS")
        print()
        
        confused_pairs = []
        for i, true_crop in enumerate(crops):
            for j, pred_crop in enumerate(crops):
                if i != j and cm[i, j] > 0:  # Off-diagonal (errors)
                    confused_pairs.append({
                        'true_crop': true_crop,
                        'predicted_as': pred_crop,
                        'count': cm[i, j],
                        'pair': f"{true_crop} → {pred_crop}"
                    })
        
        # Sort by count
        confused_pairs_df = pd.DataFrame(confused_pairs).sort_values('count', ascending=False)
        
        print("\n", confused_pairs_df.head(10).to_string(index=False))
        
        # Calculate per-crop error rates
        print()
        print("PER-CROP ERROR ANALYSIS")
        print()
        
        crop_analysis = []
        for i, crop in enumerate(crops):
            total = cm[i, :].sum()
            correct = cm[i, i]
            errors = total - correct
            accuracy = (correct / total * 100) if total > 0 else 0
            
            crop_analysis.append({
                'crop': crop,
                'total_samples': total,
                'correct': correct,
                'errors': errors,
                'accuracy': accuracy,
                'error_rate': 100 - accuracy
            })
        
        crop_analysis_df = pd.DataFrame(crop_analysis).sort_values('errors', ascending=False)
        
        print("\nCrops with most errors:")
        print(crop_analysis_df.head(10).to_string(index=False))
        
        # Visualize confusion matrix
        self._plot_confusion_matrix(cm, crops)
        
        # Visualize focused confusion matrix (only problematic crops)
        problematic_crops = crop_analysis_df.head(5)['crop'].tolist()
        self._plot_focused_confusion_matrix(cm_df, problematic_crops)
        
        return cm_df, confused_pairs_df, crop_analysis_df
    
    def _plot_confusion_matrix(self, cm, crops):
        """Plot full confusion matrix"""
        plt.figure(figsize=(16, 14))
        
        # Normalize to show percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=False,  # Too crowded with 22x22
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=crops,
            yticklabels=crops,
            cbar_kws={'label': 'Proportion'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title('Confusion Matrix - All Crops (Normalized)', 
                fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Crop', fontsize=12)
        plt.xlabel('Predicted Crop', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('results/figures/confusion_matrix_full.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Full confusion matrix saved: results/figures/confusion_matrix_full.png")
        plt.show()

    def _plot_focused_confusion_matrix(self, cm_df, problematic_crops):
        """Plot focused confusion matrix for problematic crops only"""
        
        # Extract submatrix for problematic crops
        cm_focused = cm_df.loc[problematic_crops, problematic_crops]
        
        # Normalize
        cm_focused_norm = cm_focused.div(cm_focused.sum(axis=1), axis=0)
        
        plt.figure(figsize=(10, 8))
        
        # Plot with annotations
        sns.heatmap(
            cm_focused_norm,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn_r',
            xticklabels=problematic_crops,
            yticklabels=problematic_crops,
            cbar_kws={'label': 'Proportion'},
            linewidths=2,
            linecolor='white',
            vmin=0,
            vmax=1
        )
        
        plt.title('Focused Confusion Matrix - Most Problematic Crops', 
                fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Crop', fontsize=12)
        plt.xlabel('Predicted Crop', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('results/figures/confusion_matrix_focused.png', dpi=300, bbox_inches='tight')
        print(f"✅ Focused confusion matrix saved: results/figures/confusion_matrix_focused.png")
        plt.show()
    
    #=================confusion trix above--------------------------------------
    
    #===========================feature level error analysis=========================
    
    def analyze_errors_by_features(self):
        """
    Analyze if errors occur at specific feature ranges
    E.g., do errors cluster at high N, low rainfall, etc.?
    """
        print("\n" + "="*60)
        print("STEP 3: FEATURE-LEVEL ERROR ANALYSIS")
        print("="*60)
    
        if self.predictions_df is None:
            print("❌ No predictions available. Run generate_predictions() first.")
            return
    
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Compare feature distributions: errors vs correct predictions
        errors = self.predictions_df[self.predictions_df['is_correct'] == 0]
        correct = self.predictions_df[self.predictions_df['is_correct'] == 1]
        
        print(f"\nComparing feature distributions:")
        print(f"  Errors: {len(errors)} samples")
        print(f"  Correct: {len(correct)} samples")
    
        # Statistical comparison
        print()
        print("FEATURE STATISTICS COMPARISON")
        print()
    
        comparison = []
        for feature in features:
            comparison.append({
                'Feature': feature,
                'Error_Mean': errors[feature].mean(),
                'Correct_Mean': correct[feature].mean(),
                'Error_Std': errors[feature].std(),
                'Correct_Std': correct[feature].std(),
                'Difference': errors[feature].mean() - correct[feature].mean()
            })
        
        comparison_df = pd.DataFrame(comparison)
        print("\n", comparison_df.to_string(index=False))
        
        # Visualize distributions
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
        
            # Plot distributions
            ax.hist(correct[feature], bins=30, alpha=0.6, label='Correct', 
                    color='green', edgecolor='black')
            ax.hist(errors[feature], bins=30, alpha=0.6, label='Errors', 
                    color='red', edgecolor='black')
            
            ax.set_xlabel(feature, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add mean lines
            ax.axvline(correct[feature].mean(), color='green', linestyle='--', 
                    linewidth=2, label=f'Correct Mean: {correct[feature].mean():.1f}')
            ax.axvline(errors[feature].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Error Mean: {errors[feature].mean():.1f}')
        
    # Remove extra subplots
        for idx in range(len(features), 9):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Feature Distributions: Errors vs Correct Predictions', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('results/figures/feature_error_distributions.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Feature distributions saved: results/figures/feature_error_distributions.png")
        plt.show()
    
        # Analyze specific problematic crops
        print()
        print("FEATURE ANALYSIS FOR PROBLEMATIC CROPS")
        print()
        
        # Focus on groundnuts and mothbeans
        groundnuts_errors = errors[errors['true_crop'] == 'groundnuts']
        mothbeans_errors = errors[errors['true_crop'] == 'mothbeans']
        
        if len(groundnuts_errors) > 0:
            print("\nGroundnuts errors - average features:")
            for feature in features:
                print(f"  {feature}: {groundnuts_errors[feature].mean():.2f}")
        
        if len(mothbeans_errors) > 0:
            print("\nMothbeans errors - average features:")
            for feature in features:
                print(f"  {feature}: {mothbeans_errors[feature].mean():.2f}")
        
        return comparison_df

#---------------------------------feature level error analysis above--------------

#====================confidence threshold analysis===============================

    def analyze_confidence_thresholds(self):
        """
        Determine optimal confidence threshold for production use
        """
        print()
        print("STEP 4: CONFIDENCE THRESHOLD ANALYSIS")
        print()
        
        if self.predictions_df is None:
            print("❌ No predictions available. Run generate_predictions() first.")
            return
        
        # Test different confidence thresholds
        thresholds = [50, 60, 70, 75, 80, 85, 90, 95]
        
        threshold_analysis = []
        
        for threshold in thresholds:
            # Filter predictions above threshold
            above_threshold = self.predictions_df[
                self.predictions_df['confidence'] >= threshold
            ]
            
            if len(above_threshold) > 0:
                accuracy = (above_threshold['is_correct'].sum() / len(above_threshold)) * 100
                coverage = (len(above_threshold) / len(self.predictions_df)) * 100
                
                threshold_analysis.append({
                    'Threshold': f'{threshold}%',
                    'Predictions': len(above_threshold),
                    'Coverage': coverage,
                    'Accuracy': accuracy,
                    'Errors': len(above_threshold) - above_threshold['is_correct'].sum()
                })
        
        threshold_df = pd.DataFrame(threshold_analysis)
        
        print("\nConfidence Threshold Analysis:")
        print("(What happens if we only trust predictions above X% confidence?)")
        print("\n", threshold_df.to_string(index=False))
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy vs Threshold
        axes[0].plot(thresholds, threshold_df['Accuracy'], marker='o', 
                    linewidth=2, markersize=8, color='green')
        axes[0].axhline(y=93.24, color='red', linestyle='--', 
                    label='Overall Accuracy (93.24%)')
        axes[0].set_xlabel('Confidence Threshold (%)', fontsize=12)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Accuracy vs Confidence Threshold', 
                        fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim(90, 101)
        
        # Plot 2: Coverage vs Threshold
        axes[1].plot(thresholds, threshold_df['Coverage'], marker='s', 
                    linewidth=2, markersize=8, color='blue')
        axes[1].set_xlabel('Confidence Threshold (%)', fontsize=12)
        axes[1].set_ylabel('Coverage (% of predictions)', fontsize=12)
        axes[1].set_title('Coverage vs Confidence Threshold', 
                        fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig('results/figures/confidence_threshold_analysis.png', 
                dpi=300, bbox_inches='tight')
        print(f"\n✅ Threshold analysis saved: results/figures/confidence_threshold_analysis.png")
        plt.show()
        
        # Recommendation
        print()
        print("RECOMMENDATION FOR PRODUCTION USE")
        print()
        
        # Find threshold with >99% accuracy
        high_accuracy = threshold_df[threshold_df['Accuracy'] >= 99]
        
        if len(high_accuracy) > 0:
            recommended = high_accuracy.iloc[0]
            print(f"\n✅ Recommended threshold: {recommended['Threshold']}")
            print(f"   Accuracy: {recommended['Accuracy']:.2f}%")
            print(f"   Coverage: {recommended['Coverage']:.2f}% of predictions")
            print(f"   Expected errors: {recommended['Errors']:.0f} out of {recommended['Predictions']:.0f}")
            print(f"\n   Interpretation: If we only trust predictions with >{recommended['Threshold']}")
            print(f"   confidence, we'll be correct {recommended['Accuracy']:.2f}% of the time,")
            print(f"   covering {recommended['Coverage']:.2f}% of all cases.")
        
        return threshold_df
        
    #==================================threshold analysis above==========================

    #========================Error case studies==================================
    def detailed_error_case_studies(self):
        """
        Deep dive into specific error patterns
        """
        print("\n" + "="*60)
        print("STEP 5: DETAILED ERROR CASE STUDIES")
        print("="*60)
        
        if self.predictions_df is None:
            print("❌ No predictions available. Run generate_predictions() first.")
            return
        
        errors = self.predictions_df[self.predictions_df['is_correct'] == 0]
        
        # Case Study 1: Groundnuts vs Mothbeans confusion
        print("\n" + "-"*60)
        print("CASE STUDY 1: GROUNDNUTS ↔ MOTHBEANS CONFUSION")
        print("-"*60)
        
        groundnuts_as_mothbeans = errors[
            (errors['true_crop'] == 'groundnuts') & 
            (errors['predicted_crop'] == 'mothbeans')
        ]
        
        mothbeans_as_groundnuts = errors[
            (errors['true_crop'] == 'mothbeans') & 
            (errors['predicted_crop'] == 'groundnuts')
        ]
        
        print(f"\nGroundnuts misclassified as Mothbeans: {len(groundnuts_as_mothbeans)}")
        print(f"Mothbeans misclassified as Groundnuts: {len(mothbeans_as_groundnuts)}")
        print(f"Total confusion: {len(groundnuts_as_mothbeans) + len(mothbeans_as_groundnuts)}")
        
        # Compare their feature requirements
        print("\n📊 Comparing typical requirements:")
        
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Get actual data for each crop (all samples, not just errors)
        groundnuts_all = self.df[self.df['label'] == 'groundnuts']
        mothbeans_all = self.df[self.df['label'] == 'mothbeans']
        
        print("\nGroundnuts (typical):")
        for feature in features:
            print(f"  {feature}: {groundnuts_all[feature].mean():.2f} ± {groundnuts_all[feature].std():.2f}")
        
        print("\nMothbeans (typical):")
        for feature in features:
            print(f"  {feature}: {mothbeans_all[feature].mean():.2f} ± {mothbeans_all[feature].std():.2f}")
        
        print("\n🔍 Overlap analysis:")
        overlap_metrics = []
        for feature in features:
            # Calculate range overlap
            gn_min, gn_max = groundnuts_all[feature].min(), groundnuts_all[feature].max()
            mb_min, mb_max = mothbeans_all[feature].min(), mothbeans_all[feature].max()
            
            overlap_start = max(gn_min, mb_min)
            overlap_end = min(gn_max, mb_max)
            
            if overlap_start < overlap_end:
                gn_range = gn_max - gn_min
                mb_range = mb_max - mb_min
                overlap = overlap_end - overlap_start
                overlap_pct = (overlap / ((gn_range + mb_range) / 2)) * 100
            else:
                overlap_pct = 0
            
            overlap_metrics.append({
                'Feature': feature,
                'Groundnuts_Mean': groundnuts_all[feature].mean(),
                'Mothbeans_Mean': mothbeans_all[feature].mean(),
                'Difference': abs(groundnuts_all[feature].mean() - mothbeans_all[feature].mean()),
                'Overlap_%': overlap_pct
            })
        
        overlap_df = pd.DataFrame(overlap_metrics)
        print("\n", overlap_df.to_string(index=False))
        
        print("\n💡 Insights:")
        high_overlap = overlap_df[overlap_df['Overlap_%'] > 80]
        print(f"   Features with high overlap (>80%): {', '.join(high_overlap['Feature'].tolist())}")
        print("   → These features can't distinguish between the two crops well")
        
        # Visualize the confusion
        self._plot_crop_pair_comparison('groundnuts', 'mothbeans')
        
        return overlap_df
        
    def _plot_crop_pair_comparison(self, crop1, crop2):
        """Plot feature comparison for two confused crops"""
        
        crop1_data = self.df[self.df['label'] == crop1]
        crop2_data = self.df[self.df['label'] == crop2]
        
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            ax.hist(crop1_data[feature], bins=20, alpha=0.6, label=crop1, 
                    color='orange', edgecolor='black')
            ax.hist(crop2_data[feature], bins=20, alpha=0.6, label=crop2, 
                    color='purple', edgecolor='black')
            
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # Remove extra subplot
        fig.delaxes(axes[7])
        
        plt.suptitle(f'{crop1.title()} vs {crop2.title()} - Feature Overlap', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'results/figures/crop_comparison_{crop1}_vs_{crop2}.png', 
                dpi=300, bbox_inches='tight')
        print(f"\n✅ Crop comparison saved: results/figures/crop_comparison_{crop1}_vs_{crop2}.png")
        plt.show()


    #====================================learning curves============================================
    def generate_learning_curves(self):
        """
        Generate learning curves to analyze:
        1. How performance improves with more training data
        2. Whether model is overfitting or underfitting
        3. If more data would help
        """
        print()
        print("STEP 6: LEARNING CURVES ANALYSIS")
        print()
        
        from sklearn.model_selection import learning_curve
        
        # Prepare data
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = self.df[features]
        y = self.encoder.transform(self.df['label'])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        print("\nGenerating learning curves (this may take a few minutes)...")
        
        # Define training sizes to test
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Generate learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            self.model,
            X_scaled,
            y,
            train_sizes=train_sizes,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        # Calculate mean and std
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        print("\n✅ Learning curves generated")
        
        # Print results table
        print()
        print("LEARNING CURVE DATA")
        print()
        
        learning_data = []
        for i, size in enumerate(train_sizes_abs):
            learning_data.append({
                'Training_Size': int(size),
                'Training_Accuracy': f"{train_mean[i]*100:.2f}%",
                'Validation_Accuracy': f"{val_mean[i]*100:.2f}%",
                'Gap': f"{(train_mean[i] - val_mean[i])*100:.2f}%"
            })
        
        learning_df = pd.DataFrame(learning_data)
        print("\n", learning_df.to_string(index=False))
        
        # Analyze results
        print()
        print("LEARNING CURVE ANALYSIS")
        print()
        
        final_train_acc = train_mean[-1] * 100
        final_val_acc = val_mean[-1] * 100
        gap = (train_mean[-1] - val_mean[-1]) * 100
        
        print(f"\nWith full training data ({int(train_sizes_abs[-1])} samples):")
        print(f"  Training accuracy: {final_train_acc:.2f}%")
        print(f"  Validation accuracy: {final_val_acc:.2f}%")
        print(f"  Train-Val gap: {gap:.2f}%")
        
        # Interpretation
        print("\n💡 Interpretation:")
        if gap < 2:
            print("   ✅ Low bias, low variance - Model is well-balanced")
            print("   → More data unlikely to significantly improve performance")
        elif gap < 5:
            print("   ✅ Slight overfitting but acceptable")
            print("   → Current model is performing well")
        elif gap < 10:
            print("   ⚠️ Moderate overfitting detected")
            print("   → Consider: regularization, more data, or simpler model")
        else:
            print("   ❌ High overfitting")
            print("   → Action needed: Add more data, reduce complexity, or add regularization")
        
        if val_mean[-1] < 0.85:
            print("   ⚠️ Low validation accuracy - High bias (underfitting)")
            print("   → Consider: More complex model or better features")
        
        # Check if curves are converging
        val_improvement = val_mean[-1] - val_mean[-2]
        if val_improvement > 0.01:
            print("   📈 Validation accuracy still improving")
            print("   → More training data might help")
        else:
            print("   📊 Validation accuracy has plateaued")
            print("   → More data unlikely to help significantly")
        
        # Visualize
        self._plot_learning_curves(train_sizes_abs, train_mean, train_std, 
                                val_mean, val_std)
        
        return learning_df

    def _plot_learning_curves(self, train_sizes, train_mean, train_std, 
                            val_mean, val_std):
        """Plot learning curves"""
        
        plt.figure(figsize=(12, 6))
        
        # Plot training scores
        plt.plot(train_sizes, train_mean, 'o-', color='blue', 
                label='Training accuracy', linewidth=2, markersize=8)
        plt.fill_between(train_sizes, 
                        train_mean - train_std, 
                        train_mean + train_std, 
                        alpha=0.2, color='blue')
        
        # Plot validation scores
        plt.plot(train_sizes, val_mean, 'o-', color='green', 
                label='Validation accuracy (5-fold CV)', linewidth=2, markersize=8)
        plt.fill_between(train_sizes, 
                        val_mean - val_std, 
                        val_mean + val_std, 
                        alpha=0.2, color='green')
        
        # Add reference line
        plt.axhline(y=0.9324, color='red', linestyle='--', 
                label='Test accuracy (93.24%)', linewidth=2)
        
        plt.xlabel('Training Set Size (samples)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Learning Curves - Model Performance vs Training Size', 
                fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0.7, 1.02)
        
        # Add annotations
        final_gap = (train_mean[-1] - val_mean[-1]) * 100
        plt.annotate(f'Final gap: {final_gap:.2f}%',
                    xy=(train_sizes[-1], train_mean[-1]),
                    xytext=(train_sizes[-1]*0.7, train_mean[-1]-0.05),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/figures/learning_curves.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Learning curves saved: results/figures/learning_curves.png")
        plt.show()

    def analyze_per_crop_performance(self):
        """
        Analyze which crops need more training data
        """
        print()
        print("PER-CROP PERFORMANCE ANALYSIS")
        print()
        
        if self.predictions_df is None:
            print("❌ No predictions available. Run generate_predictions() first.")
            return
        
        # Get per-crop accuracy from predictions
        crop_performance = []
        
        for crop in sorted(self.df['label'].unique()):
            # Get test samples for this crop
            crop_preds = self.predictions_df[self.predictions_df['true_crop'] == crop]
            
            if len(crop_preds) > 0:
                accuracy = (crop_preds['is_correct'].sum() / len(crop_preds)) * 100
                avg_confidence = crop_preds['confidence'].mean()
                
                # Get training set size for this crop
                train_size = len(self.df[self.df['label'] == crop]) * 0.8  # Approximate
                
                crop_performance.append({
                    'Crop': crop,
                    'Test_Samples': len(crop_preds),
                    'Correct': crop_preds['is_correct'].sum(),
                    'Accuracy': accuracy,
                    'Avg_Confidence': avg_confidence,
                    'Est_Train_Size': int(train_size)
                })
        
        performance_df = pd.DataFrame(crop_performance).sort_values('Accuracy')
        
        print()
        print("CROPS RANKED BY PERFORMANCE (Worst to Best)")
        print()
        print("\n", performance_df.to_string(index=False))
        
        # Identify problematic crops
        poor_performers = performance_df[performance_df['Accuracy'] < 90]
        
        if len(poor_performers) > 0:
            print("\n⚠️ Crops with <90% accuracy:")
            for _, row in poor_performers.iterrows():
                print(f"   {row['Crop']}: {row['Accuracy']:.1f}% "
                    f"(trained on ~{row['Est_Train_Size']} samples)")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Accuracy by crop
        performance_df.plot(x='Crop', y='Accuracy', kind='barh', 
                        ax=axes[0], color='steelblue', legend=False)
        axes[0].axvline(x=90, color='red', linestyle='--', 
                    label='90% threshold', linewidth=2)
        axes[0].axvline(x=93.24, color='green', linestyle='--', 
                    label='Overall accuracy', linewidth=2)
        axes[0].set_xlabel('Accuracy (%)', fontsize=12)
        axes[0].set_ylabel('Crop', fontsize=12)
        axes[0].set_title('Per-Crop Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].set_xlim(0, 100)
        axes[0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Accuracy vs Training Size
        axes[1].scatter(performance_df['Est_Train_Size'], 
                    performance_df['Accuracy'],
                    s=100, alpha=0.6, color='purple')
        
        # Add labels for poor performers
        for _, row in poor_performers.iterrows():
            axes[1].annotate(row['Crop'], 
                            (row['Est_Train_Size'], row['Accuracy']),
                            fontsize=9, ha='right')
        
        axes[1].axhline(y=90, color='red', linestyle='--', 
                    label='90% threshold', linewidth=2)
        axes[1].set_xlabel('Estimated Training Samples', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Accuracy vs Training Data Size', 
                        fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig('results/figures/per_crop_performance.png', 
                dpi=300, bbox_inches='tight')
        print(f"\n✅ Per-crop performance saved: results/figures/per_crop_performance.png")
        plt.show()
        
        return performance_df 

#================= learning curves above====================================

def main():
    """Main execution function - Complete Error Analysis"""
    print()
    print("CROP RECOMMENDATION SYSTEM - COMPLETE ERROR ANALYSIS")
    print()
    
    # Initialize analyzer
    analyzer = ErrorAnalyzer()
    
    # Step 1: Generate predictions
    print()
    predictions_df = analyzer.generate_predictions()
    
    # Step 2: Error summary
    print()
    errors = analyzer.get_error_summary()
    
    # Step 3: Analyze by confidence
    print()
    confidence_analysis = analyzer.analyze_errors_by_confidence()
    
    # Step 4: Display sample errors
    print()
    sample_errors = analyzer.display_sample_errors(n=5)
    
    # Step : Confusion Matrix
    print()
    cm_df, confused_pairs_df, crop_analysis_df = analyzer.analyze_confusion_matrix()
    
    # Step : Feature-Level Analysis
    print()
    feature_comparison = analyzer.analyze_errors_by_features()
    
    # Step : Confidence Threshold
    print()
    threshold_analysis = analyzer.analyze_confidence_thresholds()
    
    # Step : Case Studies
    print()
    overlap_analysis = analyzer.detailed_error_case_studies()
    
    # Step 6: Learning Curves 
    print()
    learning_data = analyzer.generate_learning_curves()
    
    # Step 7: Per-Crop Performance
    print()
    crop_performance = analyzer.analyze_per_crop_performance()
    
    # Final Summary    
    print("📁 Generated Files:")
    print("   - results/predictions_with_errors.csv")
    print("   - results/figures/error_by_confidence.png")
    print("   - results/figures/confusion_matrix_full.png")
    print("   - results/figures/confusion_matrix_focused.png")
    print("   - results/figures/feature_error_distributions.png")
    print("   - results/figures/confidence_threshold_analysis.png")
    print("   - results/figures/crop_comparison_groundnuts_vs_mothbeans.png")
    
    # print("\n📊 Key Findings:")
    # print("   1. Model achieves 93.24% accuracy overall")
    # print("   2. Groundnuts ↔ Mothbeans confusion accounts for 87% of errors")
    # print("   3. High confidence (>85%) predictions are 99%+ accurate")
    # print("   4. Feature overlap between problematic crops explains confusion")
    
    # print("\n💡 Recommendations:")
    # print("   1. Set confidence threshold at 85% for production use")
    # print("   2. Add warning for groundnuts/mothbeans predictions")
    # print("   3. Consider collecting more discriminative features")
    # print("   4. Investigate soil texture or growth period data")


if __name__ == "__main__":
    main()


# def main():
#     """Main execution function"""
#     print()
#     print("CROP RECOMMENDATION SYSTEM - ERROR ANALYSIS")
#     print()
    
#     # Initialize analyzer
#     analyzer = ErrorAnalyzer()
    
#     # Step 1: Generate predictions
#     predictions_df = analyzer.generate_predictions()
    
#     # Step 2: Get error summary
#     errors = analyzer.get_error_summary()
    
#     # Step 3: Analyze by confidence
#     confidence_analysis = analyzer.analyze_errors_by_confidence()
    
#     # Step 4: Display sample errors
#     sample_errors = analyzer.display_sample_errors(n=10)
    
#     # print("\n" + "🔍"*30)
#     # print("ERROR ANALYSIS COMPLETE - STEP 1")
#     # print("🔍"*30 + "\n")
    
#     # print("✅ Next steps:")
#     # print("   - Review error patterns in results/predictions_with_errors.csv")
#     # print("   - Examine confidence vs accuracy relationship")
#     # print("   - Proceed to Step 2: Confusion Matrix Analysis")


# if __name__ == "__main__":
#     main()