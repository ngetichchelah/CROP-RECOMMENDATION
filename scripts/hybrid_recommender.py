
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import SimpleCF from the standalone module
try:
    from scripts.cf_model import SimpleCF
    CF_AVAILABLE = True
    print("✅ CF model module available")
except ImportError as e:
    print(f"⚠️  CF model module not available - CF will be disabled: {e}")
    CF_AVAILABLE = False
    
class HybridRecommender:
    """
    Hybrid recommendation system combining:
    - Content-based filtering (SVM model)
    - Collaborative filtering (SimpleCF)
    """
    
    def __init__(self, cf_model_path='models/cf_model.pkl'):
        self.cf_model_path = cf_model_path
        self.cf_model = self._load_cf_model()
        self.svm_model = self._load_svm_model()
        self.scaler = self._load_scaler()
        self.label_encoder = self._load_label_encoder()
        
        print("🤖 Hybrid Recommender Initialized")
        if self.cf_model:
            print("   ✅ Collaborative Filtering: ACTIVE")
        else:
            print("   ⚠️  Collaborative Filtering: INACTIVE (using SVM only)")
    
    def _load_cf_model(self):
        """Temporarily disable CF to avoid serialization issues"""
        print("⚠️  CF temporarily disabled - using SVM only")
        return None
    
    def _load_svm_model(self):
        """Load SVM model for content-based recommendations"""
        try:
            model_path = 'models/crop_model_svm.pkl'
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                print("✅ SVM model loaded")
                return model
            else:
                print("❌ SVM model not found")
                return None
        except Exception as e:
            print(f"❌ Error loading SVM model: {e}")
            return None
    
    def _load_scaler(self):
        """Load feature scaler"""
        try:
            scaler_path = 'models/scaler.pkl'
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print("✅ Scaler loaded")
                return scaler
            else:
                print("❌ Scaler not found")
                return None
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            return None
    
    def _load_label_encoder(self):
        """Load label encoder for crop names"""
        try:
            encoder_path = 'models/label_encoder.pkl'
            if os.path.exists(encoder_path):
                encoder = joblib.load(encoder_path)
                print("✅ Label encoder loaded")
                return encoder
            else:
                print("❌ Label encoder not found")
                return None
        except Exception as e:
            print(f"❌ Error loading label encoder: {e}")
            return None
    
    def _load_user_history(self):
        """Load user interaction history"""
        try:
            history_file = 'data/user_interactions.csv'
            if os.path.exists(history_file):
                df = pd.read_csv(history_file)
                print(f"✅ User history loaded: {len(df)} interactions")
                return df
            else:
                print("❌ User history file not found")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading user history: {e}")
            return pd.DataFrame()
    
    def get_user_history(self, user_id):
        """Get recommendation history for a specific user"""
        try:
            history_df = self._load_user_history()
            if not history_df.empty and 'user_id' in history_df.columns:
                user_history = history_df[history_df['user_id'] == user_id].to_dict('records')
                return user_history
            return []
        except Exception as e:
            print(f"Error getting user history for {user_id}: {e}")
            return []
    
    def _get_cf_score(self, user_id, crop):
        """Get CF prediction using SimpleCF"""
        # CF is temporarily disabled
        return None
    
    def _get_svm_recommendation(self, soil_params):
        """Get content-based recommendation using SVM"""
        if self.svm_model is None or self.scaler is None or self.label_encoder is None:
            return None
        
        try:
            # Prepare features
            features = np.array([[
                soil_params['N'],
                soil_params['P'], 
                soil_params['K'],
                soil_params['temperature'],
                soil_params['humidity'],
                soil_params['ph'],
                soil_params['rainfall']
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.svm_model.predict(features_scaled)
            probabilities = self.svm_model.predict_proba(features_scaled)
            
            # Get top 3 recommendations
            top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
            top_3_crops = self.label_encoder.inverse_transform(top_3_indices)
            top_3_confidences = probabilities[0][top_3_indices] * 100
            
            # Get all predictions for display
            all_predictions = []
            for i, crop in enumerate(self.label_encoder.classes_):
                confidence = probabilities[0][i] * 100
                all_predictions.append((crop, confidence))
            
            # Sort all predictions by confidence
            all_predictions.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'recommended_crop': top_3_crops[0],
                'confidence': top_3_confidences[0],
                'top_3_recommendations': list(zip(top_3_crops, top_3_confidences)),
                'all_predictions': all_predictions
            }
            
        except Exception as e:
            print(f"❌ SVM prediction failed: {e}")
            return None
    
    def recommend(self, user_id, soil_params, explain=False, constraints=None):
        """
        Get hybrid recommendation
        
        Args:
            user_id: User identifier
            soil_params: Dictionary with soil parameters
            explain: Whether to return explanation
            constraints: Resource constraints (optional)
            
        Returns:
            Dictionary with recommendation details
        """
        # Get SVM (content-based) recommendation
        svm_result = self._get_svm_recommendation(soil_params)
        
        if svm_result is None:
            return {
                'recommended_crop': 'Unknown',
                'confidence': 0,
                'method': 'error',
                'explanation': 'Model loading failed'
            }
        
        recommended_crop = svm_result['recommended_crop']
        svm_confidence = svm_result['confidence']
        method = 'svm_only'
        cf_score = None
        boosted_by_cf = False
        
        # Check if we can use collaborative filtering (currently disabled)
        user_history = self.get_user_history(user_id)
        if self.cf_model and len(user_history) > 0:
            cf_score = self._get_cf_score(user_id, recommended_crop)
            
            if cf_score is not None and cf_score > 0.6:  # CF confidence threshold
                # Boost confidence based on CF score
                boost_factor = 1.0 + (cf_score - 0.6) * 0.5  # 0-20% boost
                final_confidence = min(100, svm_confidence * boost_factor)
                method = 'hybrid'
                boosted_by_cf = True
            else:
                final_confidence = svm_confidence
                method = 'svm_only'
        else:
            # New user or no CF model
            final_confidence = svm_confidence
            method = 'svm_only'
        
        # Apply resource constraints if provided
        if constraints:
            final_recommendation = self._apply_constraints(
                svm_result, constraints, final_confidence
            )
            if final_recommendation is not None:  # FIX: Check if final_recommendation is not None
                # Update with constrained result
                recommended_crop = final_recommendation['recommended_crop']
                final_confidence = final_recommendation['confidence']
                method = final_recommendation.get('method', method)
            else:
                # No crops match constraints
                return {
                    'error': 'No crops match your resource constraints',
                    'excluded_crops': self._get_excluded_crops(svm_result, constraints),
                    'method': 'constrained'
                }
        
        # Prepare result
        result = {
            'recommended_crop': recommended_crop,
            'confidence': final_confidence,
            'method': method,
            'top_3_recommendations': svm_result['top_3_recommendations'],
            'all_predictions': svm_result['all_predictions'],
            'boosted_by_cf': boosted_by_cf
        }
        
        # Add CF details if available
        if cf_score is not None:
            result['cf_score'] = cf_score
        
        # Add explanation if requested
        if explain:
            explanation = self._generate_explanation(result, user_id, soil_params)
            result['explanation'] = explanation
        
        return result
    
    def _apply_constraints(self, svm_result, constraints, original_confidence):
        """
        Apply resource constraints to recommendations
        Simple implementation - filters crops based on constraints
        """
        try:
            # Load resource data
            resource_data = self._load_resource_data()
            if resource_data.empty:
                return None
            
            # Get top recommendations
            top_crops = [(crop, conf) for crop, conf in svm_result['all_predictions'][:10]]
            
            # Filter crops based on constraints
            feasible_crops = []
            for crop, confidence in top_crops:
                crop_info = resource_data[resource_data['crop'] == crop]
                if crop_info.empty:
                    # If no resource data, assume it's feasible
                    feasible_crops.append((crop, confidence))
                    continue
                
                crop_info = crop_info.iloc[0]
                total_cost = crop_info.get('seed_cost_usd', 0) + crop_info.get('fertilizer_cost_usd', 0)
                labor_needed = crop_info.get('labor_days', 0)
                needs_irrigation = crop_info.get('irrigation_needed', False)
                harvest_time = crop_info.get('harvest_months', 0)
                
                # Check constraints
                budget_ok = total_cost <= constraints.get('max_budget', float('inf'))
                labor_ok = labor_needed <= constraints.get('max_labor', float('inf'))
                irrigation_ok = not needs_irrigation or constraints.get('irrigation', False)
                time_ok = harvest_time <= constraints.get('max_wait', float('inf'))
                
                if budget_ok and labor_ok and irrigation_ok and time_ok:
                    feasible_crops.append((crop, confidence))
            
            if feasible_crops:
                # Return the best feasible crop
                best_crop, best_confidence = feasible_crops[0]
                return {
                    'recommended_crop': best_crop,
                    'confidence': best_confidence,
                    'method': 'constrained'
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error applying constraints: {e}")
            return None

    def _get_excluded_crops(self, svm_result, constraints):
        """
        Get list of crops excluded by constraints with reasons
        """
        try:
            resource_data = self._load_resource_data()
            if resource_data.empty:
                return []
            
            top_crops = [(crop, conf) for crop, conf in svm_result['all_predictions'][:5]]
            excluded = []
            
            for crop, confidence in top_crops:
                crop_info = resource_data[resource_data['crop'] == crop]
                if crop_info.empty:
                    continue
                
                crop_info = crop_info.iloc[0]
                total_cost = crop_info.get('seed_cost_usd', 0) + crop_info.get('fertilizer_cost_usd', 0)
                labor_needed = crop_info.get('labor_days', 0)
                needs_irrigation = crop_info.get('irrigation_needed', False)
                harvest_time = crop_info.get('harvest_months', 0)
                
                reasons = []
                if total_cost > constraints.get('max_budget', 0):
                    reasons.append(f"Cost ${total_cost} > budget ${constraints.get('max_budget', 0)}")
                if labor_needed > constraints.get('max_labor', 0):
                    reasons.append(f"Labor {labor_needed} days > available {constraints.get('max_labor', 0)} days")
                if needs_irrigation and not constraints.get('irrigation', False):
                    reasons.append("Needs irrigation (unavailable)")
                if harvest_time > constraints.get('max_wait', 0):
                    reasons.append(f"Harvest {harvest_time} months > wait limit {constraints.get('max_wait', 0)} months")
                
                if reasons:
                    excluded.append((crop, confidence, reasons))
            
            return excluded
            
        except Exception as e:
            print(f"Error getting excluded crops: {e}")
            return []

    def _load_resource_data(self):
        """Load crop resource requirements"""
        try:
            resource_file = 'data/crop_resources.csv'
            if os.path.exists(resource_file):
                return pd.read_csv(resource_file)
            else:
                # Return default data if file doesn't exist
                return pd.DataFrame({
                    'crop': ['rice', 'maize', 'chickpea', 'cotton', 'coffee'],
                    'seed_cost_usd': [30, 15, 20, 40, 60],
                    'fertilizer_cost_usd': [90, 45, 20, 110, 140],
                    'labor_days': [80, 40, 30, 120, 100],
                    'irrigation_needed': [True, False, False, False, True],
                    'harvest_months': [4, 4, 3, 6, 36],
                    'market_access': ['HIGH', 'HIGH', 'MEDIUM', 'MEDIUM', 'LOW']
                })
        except Exception as e:
            print(f"Error loading resource data: {e}")
            return pd.DataFrame()
    
    def _generate_explanation(self, result, user_id, soil_params):
        """Generate human-readable explanation for the recommendation"""
        crop = result['recommended_crop']
        confidence = result['confidence']
        method = result['method']
        
        explanations = {
            'svm_only': f"Based on soil analysis, {crop} is the best match for your farm conditions.",
            'hybrid': f"Based on both soil analysis and preferences of farmers similar to you, {crop} is recommended.",
        }
        
        base_explanation = explanations.get(method, explanations['svm_only'])
        
        if method == 'hybrid' and 'cf_score' in result:
            cf_percentage = result['cf_score'] * 100
            base_explanation += f" Farmers with similar preferences rated this crop highly ({cf_percentage:.1f}% match)."
        
        return base_explanation
    
    def is_cf_active(self):
        """Check if collaborative filtering is active"""
        return self.cf_model is not None

# For testing
if __name__ == "__main__":
    # Test the recommender
    soil_test = {
        'N': 45, 'P': 50, 'K': 60,
        'temperature': 25, 'humidity': 60, 
        'ph': 6.5, 'rainfall': 100
    }
    
    recommender = HybridRecommender()
    result = recommender.recommend('FARMER_KE_001', soil_test, explain=True)
    
    print("\n🧪 Test Recommendation:")
    print(f"Crop: {result['recommended_crop']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"Method: {result['method']}")
    if 'cf_score' in result:
        print(f"CF Score: {result['cf_score']:.2f}")
    print(f"Explanation: {result.get('explanation', 'N/A')}")

# import pandas as pd
# import numpy as np
# import joblib
# import os
# from sklearn.preprocessing import LabelEncoder
# import sys
# import warnings
# warnings.filterwarnings('ignore')

# # Add the project root to Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# # Try to import SimpleCF from the standalone module
# try:
#     from scripts.cf_model import SimpleCF
#     CF_AVAILABLE = True
# except ImportError:
#     print("⚠️  CF model module not available - CF will be disabled")
#     CF_AVAILABLE = False
    
# class HybridRecommender:
#     """
#     Hybrid recommendation system combining:
#     - Content-based filtering (SVM model)
#     - Collaborative filtering (Implicit ALS)
#     """
    
#     def __init__(self, cf_model_path='models/cf_model.pkl'):
#         self.cf_model_path = cf_model_path
#         self.cf_model = self._load_cf_model()
#         self.svm_model = self._load_svm_model()
#         self.scaler = self._load_scaler()
#         self.label_encoder = self._load_label_encoder()
#         self.user_history = self._load_user_history()
        
#         print("🤖 Hybrid Recommender Initialized")
#         if self.cf_model:
#             print("   ✅ Collaborative Filtering: ACTIVE")
#         else:
#             print("   ⚠️  Collaborative Filtering: INACTIVE (using SVM only)")
    
#     # def _load_cf_model(self):
#     #     """Load Implicit CF model"""
#     #     try:
#     #         if os.path.exists(self.cf_model_path):
#     #             model_package = joblib.load(self.cf_model_path)
#     #             print(f"✅ CF model loaded: {len(model_package['user_ids'])} users, {len(model_package['crop_names'])} crops")
#     #             return model_package
#     #         else:
#     #             print("❌ CF model file not found")
#     #             return None
#     #     except Exception as e:
#     #         print(f"❌ Error loading CF model: {e}")
#     #         return None
    
#     def _load_cf_model(self):
#         """Load Simple CF model with proper error handling"""
#         if not CF_AVAILABLE:
#             print("❌ CF module not available")
#             return None
        
#         try:
#             if os.path.exists(self.cf_model_path):
#                 model_package = joblib.load(self.cf_model_path)
#                 model = model_package['model_object']
#                 metadata = model_package['metadata', {}]
                
#                 print(f"✅ Simple CF model loaded: {metadata['n_users']} users, {metadata['n_crops']} crops")
#                 return model_package
#             else:
#                 print("❌ CF model file not found")
#                 return None
#         except Exception as e:
#             print(f"❌ Error loading CF model: {e}")
#             return None
    
#     def _load_svm_model(self):
#         """Load SVM model for content-based recommendations"""
#         try:
#             model_path = 'models/crop_model_svm.pkl'
#             if os.path.exists(model_path):
#                 model = joblib.load(model_path)
#                 print("✅ SVM model loaded")
#                 return model
#             else:
#                 print("❌ SVM model not found")
#                 return None
#         except Exception as e:
#             print(f"❌ Error loading SVM model: {e}")
#             return None
    
#     def _load_scaler(self):
#         """Load feature scaler"""
#         try:
#             scaler_path = 'models/scaler.pkl'
#             if os.path.exists(scaler_path):
#                 scaler = joblib.load(scaler_path)
#                 print("✅ Scaler loaded")
#                 return scaler
#             else:
#                 print("❌ Scaler not found")
#                 return None
#         except Exception as e:
#             print(f"❌ Error loading scaler: {e}")
#             return None
    
#     def _load_label_encoder(self):
#         """Load label encoder for crop names"""
#         try:
#             encoder_path = 'models/label_encoder.pkl'
#             if os.path.exists(encoder_path):
#                 encoder = joblib.load(encoder_path)
#                 print("✅ Label encoder loaded")
#                 return encoder
#             else:
#                 print("❌ Label encoder not found")
#                 return None
#         except Exception as e:
#             print(f"❌ Error loading label encoder: {e}")
#             return None
    
#     def _load_user_history(self):
#         """Load user interaction history"""
#         try:
#             history_file = 'data/user_interactions.csv'
#             if os.path.exists(history_file):
#                 df = pd.read_csv(history_file)
#                 user_history = df.groupby('user_id')['recommended_crop'].apply(list).to_dict()
#                 print(f"✅ User history loaded: {len(user_history)} users")
#                 return df
#             else:
#                 print("❌ User history file not found")
#                 return pd.DataFrame()
#         except Exception as e:
#             print(f"❌ Error loading user history: {e}")
#             return {}
    
#     def get_user_history(self, user_id):
#         """Get recommendation history for a specific user"""
#         try:
#             history_df = self._load_user_history()
#             if not history_df.empty and 'user_id' in history_df.columns:
#                 user_history = history_df[history_df['user_id'] == user_id].to_dict('records')
#                 return user_history
#             return []
#         except Exception as e:
#             print(f"Error getting user history for {user_id}: {e}")
#             return []
    
#     def _get_cf_score(self, user_id, crop):
#         """Get CF prediction using SimpleCF"""
#         if self.cf_model is None:
#             return None
        
#         try:
#             model = self.cf_model['model']
#             predicted_rating = model.predict(user_id, crop)
            
#             if predicted_rating is not None:
#                 # Convert 1-5 rating to 0-1 score
#                 normalized_score = (predicted_rating - 1) / 4.0  # Maps 1->0, 5->1
#                 return float(normalized_score)
#             else:
#                 return None
                
#         except Exception as e:
#             print(f"Warning: CF prediction failed for {user_id}, {crop}: {e}")
#             return None
    
#     # def _get_cf_score(self, user_id, crop):
#     #     """Get CF prediction using Implicit ALS"""
#     #     if self.cf_model is None:
#     #         return None
        
#     #     try:
#     #         model = self.cf_model['model']
#     #         user_to_idx = self.cf_model['user_to_idx']
#     #         crop_to_idx = self.cf_model['crop_to_idx']
            
#     #         if user_id not in user_to_idx or crop not in crop_to_idx:
#     #             return None
            
#     #         user_idx = user_to_idx[user_id]
#     #         crop_idx = crop_to_idx[crop]
            
#     #         # Get prediction using model's predict method
#     #         score = model.predict(user_idx, crop_idx)
            
#     #         # Normalize score to 0-1 range
#     #         normalized_score = 1 / (1 + np.exp(-score))
            
#     #         # Ensure it's between 0 and 1
#     #         normalized_score = max(0.0, min(1.0, normalized_score))
            
#     #         return float(normalized_score)
            
#     #     except Exception as e:
#     #         print(f"Warning: CF prediction failed for {user_id}, {crop}: {e}")
#     #         return None
    
#     def _get_svm_recommendation(self, soil_params):
#         """Get content-based recommendation using SVM"""
#         if self.svm_model is None or self.scaler is None or self.label_encoder is None:
#             return None
        
#         try:
#             # Prepare features
#             features = np.array([[
#                 soil_params['N'],
#                 soil_params['P'], 
#                 soil_params['K'],
#                 soil_params['temperature'],
#                 soil_params['humidity'],
#                 soil_params['ph'],
#                 soil_params['rainfall']
#             ]])
            
#             # Scale features
#             features_scaled = self.scaler.transform(features)
            
#             # Predict
#             prediction = self.svm_model.predict(features_scaled)
#             probabilities = self.svm_model.predict_proba(features_scaled)
            
#             # Get top 3 recommendations
#             top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
#             top_3_crops = self.label_encoder.inverse_transform(top_3_indices)
#             top_3_confidences = probabilities[0][top_3_indices] * 100
            
#             # Get all predictions for display
#             all_predictions = []
#             for i, crop in enumerate(self.label_encoder.classes_):
#                 confidence = probabilities[0][i] * 100
#                 all_predictions.append((crop, confidence))
            
#             # Sort all predictions by confidence
#             all_predictions.sort(key=lambda x: x[1], reverse=True)
            
#             return {
#                 'recommended_crop': top_3_crops[0],
#                 'confidence': top_3_confidences[0],
#                 'top_3_recommendations': list(zip(top_3_crops, top_3_confidences)),
#                 'all_predictions': all_predictions
#             }
            
#         except Exception as e:
#             print(f"❌ SVM prediction failed: {e}")
#             return None
    
#     def recommend(self, user_id, soil_params, explain=False, constraints=None):
#         """
#         Get hybrid recommendation
        
#         Args:
#             user_id: User identifier
#             soil_params: Dictionary with soil parameters
#             explain: Whether to return explanation
            
#         Returns:
#             Dictionary with recommendation details
#         """
#         # Get SVM (content-based) recommendation
#         svm_result = self._get_svm_recommendation(soil_params)
        
#         if svm_result is None:
#             return {
#                 'recommended_crop': 'Unknown',
#                 'confidence': 0,
#                 'method': 'error',
#                 'explanation': 'Model loading failed'
#             }
        
#         recommended_crop = svm_result['recommended_crop']
#         svm_confidence = svm_result['confidence']
#         method = 'svm_only'
#         cf_score = None
#         boosted_by_cf = False
        
#         # Check if we can use collaborative filtering
#         user_history = self.get_user_history(user_id)
#         if self.cf_model and len(user_history) > 0:
#             cf_score = self._get_cf_score(user_id, recommended_crop)
            
#             if cf_score is not None and cf_score > 0.6:  # CF confidence threshold
#                 # Boost confidence based on CF score
#                 boost_factor = 1.0 + (cf_score - 0.6) * 0.5  # 0-20% boost
#                 final_confidence = min(100, svm_confidence * boost_factor)
#                 method = 'hybrid'
#                 boosted_by_cf = True
#             else:
#                 final_confidence = svm_confidence
#                 method = 'svm_only'
#         else:
#             # New user or no CF model
#             final_confidence = svm_confidence
#             method = 'svm_only'
        
#         # Apply resource constraints if provided
#         if constraints:
#             final_recommendation = self._apply_constraints(
#                 svm_result, constraints, final_confidence
#         )
#         if final_recommendation:
#             # Update with constrained result
#             recommended_crop = final_recommendation['recommended_crop']
#             final_confidence = final_recommendation['confidence']
#             method = final_recommendation.get('method', method)
#         else:
#             # No crops match constraints
#             return {
#                 'error': 'No crops match your resource constraints',
#                 'excluded_crops': self._get_excluded_crops(svm_result, constraints),
#                 'method': 'constrained'
#             }
        
#         # Prepare result
#         result = {
#             'recommended_crop': recommended_crop,
#             'confidence': final_confidence,
#             'method': method,
#             'top_3_recommendations': svm_result['top_3_recommendations'],
#             'all_predictions': svm_result['all_predictions'],
#             'boosted_by_cf': boosted_by_cf
#         }
        
#         # Add CF details if available
#         if cf_score is not None:
#             result['cf_score'] = cf_score
        
#         # Add explanation if requested
#         if explain:
#             explanation = self._generate_explanation(result, user_id, soil_params)
#             result['explanation'] = explanation
        
#         return result
    
#     def _apply_constraints(self, svm_result, constraints, original_confidence):
#         """
#         Apply resource constraints to recommendations
#         Simple implementation - filters crops based on constraints
#         """
#         try:
#             # Load resource data
#             resource_data = self._load_resource_data()
#             if resource_data.empty:
#                 return None
            
#             # Get top recommendations
#             top_crops = [(crop, conf) for crop, conf in svm_result['all_predictions'][:10]]
            
#             # Filter crops based on constraints
#             feasible_crops = []
#             for crop, confidence in top_crops:
#                 crop_info = resource_data[resource_data['crop'] == crop]
#                 if crop_info.empty:
#                     # If no resource data, assume it's feasible
#                     feasible_crops.append((crop, confidence))
#                     continue
                
#                 crop_info = crop_info.iloc[0]
#                 total_cost = crop_info.get('seed_cost_usd', 0) + crop_info.get('fertilizer_cost_usd', 0)
#                 labor_needed = crop_info.get('labor_days', 0)
#                 needs_irrigation = crop_info.get('irrigation_needed', False)
#                 harvest_time = crop_info.get('harvest_months', 0)
                
#                 # Check constraints
#                 budget_ok = total_cost <= constraints.get('max_budget', float('inf'))
#                 labor_ok = labor_needed <= constraints.get('max_labor', float('inf'))
#                 irrigation_ok = not needs_irrigation or constraints.get('irrigation', False)
#                 time_ok = harvest_time <= constraints.get('max_wait', float('inf'))
                
#                 if budget_ok and labor_ok and irrigation_ok and time_ok:
#                     feasible_crops.append((crop, confidence))
            
#             if feasible_crops:
#                 # Return the best feasible crop
#                 best_crop, best_confidence = feasible_crops[0]
#                 return {
#                     'recommended_crop': best_crop,
#                     'confidence': best_confidence,
#                     'method': 'constrained'
#                 }
#             else:
#                 return None
                
#         except Exception as e:
#             print(f"Error applying constraints: {e}")
#             return None

#     def _get_excluded_crops(self, svm_result, constraints):
#         """
#         Get list of crops excluded by constraints with reasons
#         """
#         try:
#             resource_data = self._load_resource_data()
#             if resource_data.empty:
#                 return []
            
#             top_crops = [(crop, conf) for crop, conf in svm_result['all_predictions'][:5]]
#             excluded = []
            
#             for crop, confidence in top_crops:
#                 crop_info = resource_data[resource_data['crop'] == crop]
#                 if crop_info.empty:
#                     continue
                
#                 crop_info = crop_info.iloc[0]
#                 total_cost = crop_info.get('seed_cost_usd', 0) + crop_info.get('fertilizer_cost_usd', 0)
#                 labor_needed = crop_info.get('labor_days', 0)
#                 needs_irrigation = crop_info.get('irrigation_needed', False)
#                 harvest_time = crop_info.get('harvest_months', 0)
                
#                 reasons = []
#                 if total_cost > constraints.get('max_budget', 0):
#                     reasons.append(f"Cost ${total_cost} > budget ${constraints.get('max_budget', 0)}")
#                 if labor_needed > constraints.get('max_labor', 0):
#                     reasons.append(f"Labor {labor_needed} days > available {constraints.get('max_labor', 0)} days")
#                 if needs_irrigation and not constraints.get('irrigation', False):
#                     reasons.append("Needs irrigation (unavailable)")
#                 if harvest_time > constraints.get('max_wait', 0):
#                     reasons.append(f"Harvest {harvest_time} months > wait limit {constraints.get('max_wait', 0)} months")
                
#                 if reasons:
#                     excluded.append((crop, confidence, reasons))
            
#             return excluded
            
#         except Exception as e:
#             print(f"Error getting excluded crops: {e}")
#             return []

#     def _load_resource_data(self):
#         """Load crop resource requirements"""
#         try:
#             resource_file = 'data/crop_resources.csv'
#             if os.path.exists(resource_file):
#                 return pd.read_csv(resource_file)
#             else:
#                 # Return default data if file doesn't exist
#                 return pd.DataFrame({
#                     'crop': ['rice', 'maize', 'chickpea', 'cotton', 'coffee'],
#                     'seed_cost_usd': [30, 15, 20, 40, 60],
#                     'fertilizer_cost_usd': [90, 45, 20, 110, 140],
#                     'labor_days': [80, 40, 30, 120, 100],
#                     'irrigation_needed': [True, False, False, False, True],
#                     'harvest_months': [4, 4, 3, 6, 36],
#                     'market_access': ['HIGH', 'HIGH', 'MEDIUM', 'MEDIUM', 'LOW']
#                 })
#         except Exception as e:
#             print(f"Error loading resource data: {e}")
#             return pd.DataFrame()
    
#     def _generate_explanation(self, result, user_id, soil_params):
#         """Generate human-readable explanation for the recommendation"""
#         crop = result['recommended_crop']
#         confidence = result['confidence']
#         method = result['method']
        
#         explanations = {
#             'svm_only': f"Based on soil analysis, {crop} is the best match for your farm conditions.",
#             'hybrid': f"Based on both soil analysis and preferences of farmers similar to you, {crop} is recommended.",
#         }
        
#         base_explanation = explanations.get(method, explanations['svm_only'])
        
#         if method == 'hybrid' and 'cf_score' in result:
#             cf_percentage = result['cf_score'] * 100
#             base_explanation += f" Farmers with similar preferences rated this crop highly ({cf_percentage:.1f}% match)."
        
#         return base_explanation
    
#     def get_user_recommendation_history(self, user_id):
#         """Get recommendation history for a user"""
#         return self.user_history.get(user_id, [])
    
#     def is_cf_active(self):
#         """Check if collaborative filtering is active"""
#         return self.cf_model is not None

# # For testing
# if __name__ == "__main__":
#     # Test the recommender
#     soil_test = {
#         'N': 45, 'P': 50, 'K': 60,
#         'temperature': 25, 'humidity': 60, 
#         'ph': 6.5, 'rainfall': 100
#     }
    
#     recommender = HybridRecommender()
#     result = recommender.recommend('FARMER_KE_001', soil_test, explain=True)
    
#     print("\n🧪 Test Recommendation:")
#     print(f"Crop: {result['recommended_crop']}")
#     print(f"Confidence: {result['confidence']:.1f}%")
#     print(f"Method: {result['method']}")
#     if 'cf_score' in result:
#         print(f"CF Score: {result['cf_score']:.2f}")
#     print(f"Explanation: {result.get('explanation', 'N/A')}")

# # """
# # Hybrid Crop Recommender: Content-Based (SVM) + Collaborative Filtering (SVD)
# # """

# # import sys
# # import os
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # import pandas as pd
# # import joblib
# # from utils.helpers import get_user_history

# # class HybridRecommender:
# #     """
# #     Combines SVM (content-based) with CF (collaborative filtering)
# #     Falls back to pure SVM when CF unavailable or user has insufficient history
# #     """
    
# #     def __init__(self, 
# #                  svm_model_path='models/crop_model_svm.pkl',
# #                  cf_model_path='models/cf_model.pkl',
# #                  min_cf_interactions=3):
# #         """
# #         Initialize hybrid recommender
        
# #         Parameters:
# #         -----------
# #         svm_model_path : str
# #             Path to SVM model (via CropPredictor)
# #         cf_model_path : str
# #             Path to trained CF model
# #         min_cf_interactions : int
# #             Minimum interactions needed to use CF
# #         """
        
# #         # Load SVM predictor
# #         try:
# #             from src.models.predict import CropPredictor
# #             self.svm_predictor = CropPredictor(model_path=svm_model_path)
# #             print("✅ SVM predictor loaded")
# #         except Exception as e:
# #             print(f"❌ Failed to load SVM predictor: {e}")
# #             raise
        
# #         # Load CF model (optional)
# #         self.cf_model = None
# #         self.cf_available = False
        
# #         if os.path.exists(cf_model_path):
# #             try:
# #                 self.cf_model = joblib.load(cf_model_path)
# #                 self.cf_available = True
# #                 print("✅ CF model loaded (hybrid mode active)")
# #             except Exception as e:
# #                 print(f"⚠️  CF model load failed: {e}")
# #                 print("   Will use content-based (SVM) only")
# #         else:
# #             print("ℹ️  CF model not found (using content-based only)")
        
# #         self.min_cf_interactions = min_cf_interactions
        
# #         # Load resource data
# #         try:
# #             self.resource_data = pd.read_csv('data/crop_resources.csv')
# #         except:
# #             self.resource_data = None
# #             print("⚠️  Resource data not found (filtering disabled)")
    
# #     def recommend(self, user_id, soil_params, constraints=None, explain=True):
# #         """
# #         Generate hybrid recommendation
        
# #         Parameters:
# #         -----------
# #         user_id : str
# #         soil_params : dict with N, P, K, temperature, humidity, ph, rainfall
# #         constraints : dict or None
# #             If None, no resource filtering applied
# #             If dict: {max_budget, max_labor, irrigation, max_wait}
# #         explain : bool
        
# #         Returns:
# #         --------
# #         dict matching existing CropPredictor format with additions:
# #             - recommended_crop : str
# #             - confidence : float (0-100)
# #             - top_3_recommendations : list of (crop, confidence_pct)
# #             - all_predictions : list of (crop, confidence_pct)
# #             - explanations : list of strings
# #             - warnings : list of strings
# #             - scenarios : dict
# #             - method : str ('hybrid' or 'content_based')
# #             - resource_filtered : bool
# #             - excluded_crops : list (if constraints applied)
# #         """
        
# #         # === STEP 1: Get SVM predictions ===
# #         svm_result = self.svm_predictor.recommend_crop(**soil_params)
        
# #         # Extract SVM scores (convert from percentage to 0-1)
# #         svm_scores = {
# #             crop: conf / 100.0 
# #             for crop, conf in svm_result.get('all_predictions', svm_result['top_3_recommendations'])
# #         }
        
# #         # === STEP 2: Check if CF can be used ===
# #         user_history = get_user_history(user_id)
        
# #         use_cf = (
# #             self.cf_available and 
# #             user_history is not None and 
# #             len(user_history) >= self.min_cf_interactions
# #         )
        
# #         if use_cf:
# #             # === STEP 3: Get CF predictions ===
# #             cf_scores = {}
            
# #             for crop in svm_scores.keys():
# #                 try:
# #                     prediction = self.cf_model.predict(user_id, crop)
# #                     # Normalize CF rating (1-5) to 0-1 scale
# #                     cf_scores[crop] = (prediction.est - 1) / 4.0
# #                 except:
# #                     # If prediction fails, use neutral score
# #                     cf_scores[crop] = 0.5
            
# #             # === STEP 4: Combine scores (60% SVM + 40% CF) ===
# #             hybrid_scores = {
# #                 crop: 0.6 * svm_scores[crop] + 0.4 * cf_scores.get(crop, 0.5)
# #                 for crop in svm_scores.keys()
# #             }
            
# #             # Sort by hybrid score
# #             ranked_crops = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            
# #             method = "hybrid"
        
# #         else:
# #             # === Use pure SVM ===
# #             ranked_crops = sorted(svm_scores.items(), key=lambda x: x[1], reverse=True)
# #             method = "content_based"
        
# #         # === STEP 5: Apply resource constraints (ONLY if provided) ===
# #         resource_filtered = False
# #         excluded_crops = []
        
# #         if constraints is not None and self.resource_data is not None:
# #             feasible, excluded = self._filter_by_resources(ranked_crops, constraints)
            
# #             if not feasible:
# #                 # No crops match constraints
# #                 return {
# #                     'error': 'No crops match your resource constraints',
# #                     'excluded_crops': excluded,
# #                     'constraints': constraints
# #                 }
            
# #             ranked_crops = feasible
# #             excluded_crops = excluded
# #             resource_filtered = True
        
# #         # === STEP 6: Prepare final result ===
# #         top_crop, top_score = ranked_crops[0]
# #         top_3 = [(crop, score * 100) for crop, score in ranked_crops[:3]]
# #         all_preds = [(crop, score * 100) for crop, score in ranked_crops]
        
# #         # Generate explanation
# #         explanation_text = []
# #         if explain:
# #             explanation_text = self._generate_explanation(
# #                 crop=top_crop,
# #                 svm_score=svm_scores[top_crop] * 100,
# #                 method=method,
# #                 user_history=user_history
# #             )
        
# #         # Build result matching CropPredictor format
# #         result = {
# #             'recommended_crop': top_crop,
# #             'confidence': top_score * 100,
# #             'top_3_recommendations': top_3,
# #             'all_predictions': all_preds,
# #             'explanations': explanation_text,
# #             'warnings': svm_result.get('warnings', []),
# #             'scenarios': svm_result.get('scenarios', {}),
# #             'method': method,
# #             'resource_filtered': resource_filtered,
# #             'excluded_crops': excluded_crops,
# #             'user_interactions': len(user_history) if user_history else 0
# #         }
        
# #         return result
    
# #     def _filter_by_resources(self, ranked_crops, constraints):
# #         """
# #         Filter crops by resource constraints
        
# #         Parameters:
# #         -----------
# #         ranked_crops : list of (crop, score_0_1)
# #         constraints : dict with max_budget, max_labor, irrigation, max_wait
        
# #         Returns:
# #         --------
# #         (feasible_crops, excluded_crops)
# #         """
# #         feasible = []
# #         excluded = []
        
# #         for crop, score in ranked_crops:
# #             crop_info = self.resource_data[self.resource_data['crop'] == crop]
            
# #             if crop_info.empty:
# #                 # Unknown crop, allow by default
# #                 feasible.append((crop, score))
# #                 continue
            
# #             crop_info = crop_info.iloc[0]
# #             total_cost = crop_info['seed_cost_usd'] + crop_info['fertilizer_cost_usd']
# #             labor_needed = crop_info['labor_days']
# #             needs_irrigation = crop_info['irrigation_needed']
# #             harvest_time = crop_info['harvest_months']
            
# #             reasons = []
            
# #             if total_cost > constraints['max_budget']:
# #                 reasons.append(f"💰 Cost ${total_cost:.0f} > Budget ${constraints['max_budget']}")
            
# #             if labor_needed > constraints['max_labor']:
# #                 reasons.append(f"👷 Needs {labor_needed} days > {constraints['max_labor']} available")
            
# #             if needs_irrigation and not constraints['irrigation']:
# #                 reasons.append("💧 Requires irrigation (unavailable)")
            
# #             if harvest_time > constraints['max_wait']:
# #                 reasons.append(f"⏱️ Harvest {harvest_time}mo > {constraints['max_wait']}mo limit")
            
# #             if reasons:
# #                 excluded.append((crop, score * 100, reasons))
# #             else:
# #                 feasible.append((crop, score))
        
# #         return feasible, excluded
    
# #     def _generate_explanation(self, crop, svm_score, method, user_history):
# #         """Generate simple explanation"""
# #         explanations = []
        
# #         explanations.append(
# #             f"✅ Soil and climate analysis: {svm_score:.1f}% suitability"
# #         )
        
# #         if method == 'hybrid' and user_history:
# #             explanations.append(
# #                 f"✅ Validated by collaborative filtering based on {len(user_history)} interactions"
# #             )
# #         elif method == 'content_based' and user_history and len(user_history) < self.min_cf_interactions:
# #             explanations.append(
# #                 f"ℹ️  Content-based only (need {self.min_cf_interactions - len(user_history)} more interactions for personalization)"
# #             )
# #         else:
# #             explanations.append(
# #                 "ℹ️  Content-based recommendation (no interaction history yet)"
# #             )
        
# #         return explanations