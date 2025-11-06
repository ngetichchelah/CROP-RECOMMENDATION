"""
Train and evaluate multiple ML models for crop recommendation
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, 
                             recall_score)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CropRecommendationModel:
    """Class to handle crop recommendation model training and evaluation"""
    
    #def __init__(self, data_path='data/processed/crop_data_cleaned.csv'):
    def __init__(self, data_path='data/processed/crop_data_with_features.csv'):
        """Initialize with data path"""
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = None
        self.label_encoder = None
        
    def load_data(self):
        """Load and prepare data"""
        print("LOADING DATA")
        
        # Load dataset
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        
        # Separate features and target
        X = df.drop('label', axis=1)
        y = df['label']
        
        print(f"Features: {list(X.columns)}")
        print(f"Number of classes: {y.nunique()}")
        print(f"Classes: {sorted(y.unique())}")
        
        # Encode categorical features automatically
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category').cat.codes  # convert text to numbers
    
        # Encode target labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
                
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""        
        print("TRAINING RANDOM FOREST")
                
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        self.models['Random Forest'] = rf_model
        
        print("Random Forest trained successfully")
        return rf_model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""        
        print("TRAINING XGBOOST")
                
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        print("XGBoost trained successfully")
        return xgb_model
    
    def train_svm(self, X_train, y_train):
        """Train SVM model"""        
        print("TRAINING SVM")
                
        svm_model = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=42,
            probability=True
        )
        
        svm_model.fit(X_train, y_train)
        self.models['SVM'] = svm_model
        
        print("SVM trained successfully")
        return svm_model
    
    def train_knn(self, X_train, y_train):
        """Train KNN model"""        
        print("TRAINING KNN")
        
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        self.models['KNN'] = knn_model
        
        print("KNN trained successfully")
        return knn_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        return y_pred
    
    def train_all_models(self):
        """Train all models and evaluate"""
        # Load data
        X_train, X_test, y_train, y_test, feature_names = self.load_data()
        
        # Train models
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_svm(X_train, y_train)
        self.train_knn(X_train, y_train)
        
        # Evaluate all models
        print("MODEL EVALUATION")
                
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        # Return test data for visualization
        return X_test, y_test, feature_names
    
    def compare_models(self):
        """Compare all models"""
        print("MODEL COMPARISON")
                
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1_score']]
        comparison_df = comparison_df.round(4)
        
        print(comparison_df)
        
        # Find best model
        best_model_name = comparison_df['accuracy'].idxmax()
        best_accuracy = comparison_df['accuracy'].max()
        
        print(f"\nBest Model: {best_model_name}")
        print(f"   Accuracy: {best_accuracy:.4f}")
        
        return comparison_df, best_model_name
    
    def plot_model_comparison(self, comparison_df, save_path='results/figures/model_comparison.png'):
        """Plot model comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot for accuracy
        axes[0].barh(comparison_df.index, comparison_df['accuracy'], color='steelblue')
        axes[0].set_xlabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlim(0.90, 1.0)
        for i, v in enumerate(comparison_df['accuracy']):
            axes[0].text(v + 0.002, i, f'{v:.4f}', va='center')
        
        # Grouped bar plot for all metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(comparison_df.index))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            offset = (i - 1.5) * width
            axes[1].bar(x + offset, comparison_df[metric], width, 
                       label=metric.replace('_', ' ').title())
        
        axes[1].set_xlabel('Models', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(comparison_df.index, rotation=45, ha='right')
        axes[1].legend()
        axes[1].set_ylim(0.90, 1.0)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nModel comparison plot saved to: {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, y_test, model_name='Random Forest', 
                             save_path='results/figures/confusion_matrix.png'):
        """Plot confusion matrix for best model"""
        y_pred = self.results[model_name]['predictions']
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.show()
    
    def plot_feature_importance(self, feature_names, model_name='Random Forest',
                                save_path='results/figures/feature_importance.png'):
        """Plot feature importance"""
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance_df['feature'], 
                    feature_importance_df['importance'], 
                    color='teal')
            plt.xlabel('Importance', fontsize=12)
            plt.title(f'Feature Importance - {model_name}', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add percentage labels
            for i, v in enumerate(feature_importance_df['importance']):
                plt.text(v + 0.005, i, f'{v:.3f} ({v*100:.1f}%)', 
                        va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
            plt.show()
            
            print("\nFeature Importance Ranking:")
            print(feature_importance_df)
            
            return feature_importance_df
        else:
            print(f"{model_name} does not support feature importance")
    
    def cross_validate(self, model_name='Random Forest', cv=5):
        """Perform cross-validation"""
        print()
        print(f"CROSS-VALIDATION ({cv}-Fold) - {model_name}")
        print()
        
        # Load fresh data
        df = pd.read_csv(self.data_path)
        X = df.drop('label', axis=1)
        y = self.label_encoder.transform(df['label'])
        
        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category').cat.codes
                
        # Encode target labels
        y_encoded = LabelEncoder().fit_transform(y)        
        X_scaled = self.scaler.transform(X)
        
        # Perform CV
        model = self.models[model_name]
        cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
        #cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def save_models(self, best_model_name='Random Forest'):
        """Save trained models"""
        print("SAVING MODELS")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save best model
        best_model = self.models[best_model_name]
        joblib.dump(best_model, f'models/crop_model_{best_model_name.lower().replace(" ", "_")}.pkl')
        print(f"Best model saved: models/crop_model_{best_model_name.lower().replace(' ', '_')}.pkl")
        
        # Save scaler and encoder
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        print("Scaler saved: models/scaler.pkl")
        print("Label encoder saved: models/label_encoder.pkl")
        
        # Save all models
        for model_name, model in self.models.items():
            filename = f'models/crop_model_{model_name.lower().replace(" ", "_")}.pkl'
            joblib.dump(model, filename)
        
        print(f"\n All {len(self.models)} models saved successfully!")
        

def main():
    """Main execution function"""

    print("CROP RECOMMENDATION SYSTEM - MODEL TRAINING")
    
    
    # Initialize trainer
    trainer = CropRecommendationModel()
    
    # Train all models
    X_test, y_test, feature_names = trainer.train_all_models()
    
    # Compare models
    comparison_df, best_model_name = trainer.compare_models()
    
    # Plot model comparison
    trainer.plot_model_comparison(comparison_df)
    
    # Plot confusion matrix for best model
    trainer.plot_confusion_matrix(y_test, model_name=best_model_name)
    
    # Plot feature importance
    trainer.plot_feature_importance(feature_names, model_name=best_model_name)
    
    # Cross-validation
    trainer.cross_validate(model_name=best_model_name)
    
    # Save models
    trainer.save_models(best_model_name=best_model_name)
    
    print("MODEL TRAINING COMPLETE!")
   
if __name__ == "__main__":
    main()