"""
Crop Prediction Module
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class CropPredictor:
    """Crop recommendation predictor with explainability and warnings"""
    
    def __init__(self, model_path='models/crop_model_svm.pkl',
                 scaler_path='models/scaler.pkl',
                 encoder_path='models/label_encoder.pkl'):
        """Initialize predictor by loading model, scaler, and encoder"""
        
        # Load model files
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        
        # Define feature names explicitly (CRITICAL FIX)
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Load crop requirements for explanations
        try:
            self.crop_requirements = pd.read_csv('data/processed/crop_requirements_summary.csv')
        except:
            self.crop_requirements = None
    
    def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Make crop recommendation with explanations and warnings
        
        Parameters:
        -----------
        N, P, K : float
            Soil nutrient levels (kg/ha)
        temperature : float
            Temperature (°C)
        humidity : float
            Humidity (%)
        ph : float
            Soil pH level
        rainfall : float
            Rainfall (mm)
        
        Returns:
        --------
        dict with:
            - recommended_crop: str
            - confidence: float (0-100)
            - top_3_recommendations: list of (crop, confidence) tuples
            - all_predictions: list of all (crop, confidence) tuples
            - explanations: list of explanation strings
            - warnings: list of warning strings
            - scenarios: dict of scenario analysis
        """
        
        # Create input dataframe with EXACT feature order
        input_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })
        
        # Ensure column order matches training
        input_data = input_data[self.feature_names]
        
        # Scale features
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        # Get crop name and confidence
        crop_name = self.encoder.inverse_transform(prediction)[0]
        confidence = probabilities.max() * 100
        
        # Get top 3 recommendations
        top_3_idx = probabilities.argsort()[-3:][::-1]
        top_3_crops = self.encoder.inverse_transform(top_3_idx)
        top_3_probs = probabilities[top_3_idx] * 100
        top_3_recommendations = list(zip(top_3_crops, top_3_probs))
        
        # Get all predictions (for resource filtering)
        all_idx = probabilities.argsort()[::-1]
        all_crops = self.encoder.inverse_transform(all_idx)
        all_probs = probabilities[all_idx] * 100
        all_predictions = list(zip(all_crops, all_probs))
        
        # Generate explanations
        input_params = {
            'N': N, 'P': P, 'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        explanations = self.explain_prediction(input_params, crop_name)
        
        # Check for edge cases/warnings
        warnings = self.check_edge_cases(input_params)
        
        # Scenario analysis
        scenarios = self.analyze_scenarios(input_params, all_predictions)
        
        return {
            'recommended_crop': crop_name,
            'confidence': confidence,
            'top_3_recommendations': top_3_recommendations,
            'all_predictions': all_predictions,
            'explanations': explanations,
            'warnings': warnings,
            'scenarios': scenarios
        }
    
    def explain_prediction(self, input_params, crop_name):
        """
        Generate parameter-by-parameter explanations
        
        Returns list of explanation strings with ✅/⚠️ indicators
        """
        explanations = []
        
        if self.crop_requirements is None:
            return ["Crop requirements data not available for detailed explanations"]
        
        # Get ideal requirements for this crop
        crop_col = 'label' if 'label' in self.crop_requirements.columns else 'crop'
        crop_req = self.crop_requirements[self.crop_requirements[crop_col] == crop_name]
        
        if crop_req.empty:
            return [f"Requirements not found for {crop_name}"]
        
        crop_req = crop_req.iloc[0]
        
        # Check each parameter
        param_mapping = {
            'N': 'N_avg',
            'P': 'P_avg',
            'K': 'K_avg',
            'temperature': 'temp_avg',
            'humidity': 'humidity_avg',
            'ph': 'ph_avg',
            'rainfall': 'rainfall_avg'
        }
        
        param_labels = {
            'N': 'Nitrogen',
            'P': 'Phosphorus',
            'K': 'Potassium',
            'temperature': 'Temperature',
            'humidity': 'Humidity',
            'ph': 'Soil pH',
            'rainfall': 'Rainfall'
        }
        
        for param, avg_col in param_mapping.items():
            if avg_col not in crop_req.index:
                continue
            
            user_value = input_params[param]
            ideal_value = crop_req[avg_col]
            
            # Calculate percentage difference
            if ideal_value > 0:
                diff_pct = abs(user_value - ideal_value) / ideal_value * 100
            else:
                diff_pct = 0
            
            # Determine if optimal
            if diff_pct < 15:  # Within 15% is optimal
                explanations.append(
                    f"✅ {param_labels[param]}: {user_value} is optimal "
                    f"(ideal: {ideal_value:.1f})"
                )
            elif diff_pct < 30:  # 15-30% is acceptable
                explanations.append(
                    f"✓ {param_labels[param]}: {user_value} is acceptable "
                    f"(ideal: {ideal_value:.1f}, difference: {diff_pct:.0f}%)"
                )
            else:  # >30% is suboptimal
                explanations.append(
                    f"⚠️ {param_labels[param]}: {user_value} is suboptimal "
                    f"(ideal: {ideal_value:.1f}, difference: {diff_pct:.0f}%)"
                )
        
        return explanations
    
    def check_edge_cases(self, input_params):
        """
        Check for extreme/edge case conditions
        
        Returns list of warning strings
        """
        warnings = []
        
        N = input_params['N']
        P = input_params['P']
        K = input_params['K']
        temp = input_params['temperature']
        humidity = input_params['humidity']
        ph = input_params['ph']
        rainfall = input_params['rainfall']
        
        # Very high nutrients
        if N > 120:
            warnings.append(
                "⚠️ **Very High Nitrogen**: Risk of over-fertilization and nutrient runoff. "
                "Consider reducing application or choosing heavy-feeding crops."
            )
        
        if P > 120:
            warnings.append(
                "⚠️ **Very High Phosphorus**: May cause nutrient imbalance. "
                "Reduce phosphate fertilizers."
            )
        
        if K > 180:
            warnings.append(
                "⚠️ **Very High Potassium**: Excessive levels may affect other nutrient uptake."
            )
        
        # Very low nutrients
        if N < 20:
            warnings.append(
                "⚠️ **Very Low Nitrogen**: Soil needs nitrogen supplementation. "
                "Consider organic matter or fertilizer application."
            )
        
        if P < 10:
            warnings.append(
                "⚠️ **Very Low Phosphorus**: Apply phosphate fertilizer before planting."
            )
        
        if K < 10:
            warnings.append(
                "⚠️ **Very Low Potassium**: Apply potash fertilizer for better yields."
            )
        
        # Extreme temperatures
        if temp > 40:
            warnings.append(
                "⚠️ **Extreme Heat**: High temperature stress risk. "
                "Consider heat-tolerant varieties, shade netting, or mulching."
            )
        
        if temp < 10:
            warnings.append(
                "⚠️ **Cold Conditions**: Frost risk present. "
                "Delay planting or use frost protection."
            )
        
        # Drought/flood conditions
        if rainfall < 50:
            warnings.append(
                "⚠️ **Drought Conditions**: Very low rainfall. "
                "Drip irrigation and mulching strongly recommended."
            )
        
        if rainfall > 250:
            warnings.append(
                "⚠️ **Flood Risk**: Very high rainfall. "
                "Ensure proper drainage and consider raised beds."
            )
        
        # Soil pH issues
        if ph < 5.0:
            warnings.append(
                "⚠️ **Acidic Soil**: pH below 5.0. "
                "Apply lime to raise pH for most crops."
            )
        
        if ph > 8.0:
            warnings.append(
                "⚠️ **Alkaline Soil**: pH above 8.0. "
                "Apply sulfur or organic matter to lower pH."
            )
        
        return warnings
    
    def analyze_scenarios(self, input_params, all_predictions):
        """
        Analyze specific scenarios (drought, flood, etc.)
        
        Returns dict with scenario information
        """
        scenarios = {}
        
        rainfall = input_params['rainfall']
        temp = input_params['temperature']
        humidity = input_params['humidity']
        
        # Drought scenario
        if rainfall < 60:
            drought_crops = ['chickpea', 'mothbeans', 'pigeonpeas', 'groundnuts']
            alternatives = [crop for crop, conf in all_predictions if crop in drought_crops][:3]
            
            scenarios['drought'] = {
                'type': 'drought',
                'title': '🌵 Drought Conditions Detected',
                'description': f'Rainfall ({rainfall}mm) is very low. Drought-tolerant crops recommended.',
                'recommendation': 'Select crops with low water requirements.',
                'alternative_crops': alternatives,
                'advice': [
                    'Plant early to utilize residual soil moisture',
                    'Apply 10-15cm mulch layer to retain moisture',
                    'Consider drip irrigation if available',
                    'Increase plant spacing to reduce water competition',
                    'Select short-duration varieties (90-100 days)'
                ],
                'alert_level': 'warning'
            }
            scenarios['type'] = 'drought'
        
        # Flood scenario
        elif rainfall > 250:
            flood_crops = ['rice', 'jute', 'coconut']
            alternatives = [crop for crop, conf in all_predictions if crop in flood_crops][:3]
            
            scenarios['flood'] = {
                'type': 'flood',
                'title': '🌊 High Rainfall / Flood Risk',
                'description': f'Rainfall ({rainfall}mm) is very high. Water-intensive crops suitable.',
                'recommendation': 'Ensure proper drainage or select flood-tolerant crops.',
                'alternative_crops': alternatives,
                'advice': [
                    'Install drainage systems to prevent waterlogging',
                    'Use raised beds for non-water crops',
                    'Select flood-tolerant varieties',
                    'Monitor for fungal diseases (high moisture)',
                    'Consider rice or jute (thrive in wet conditions)'
                ],
                'alert_level': 'warning'
            }
            scenarios['type'] = 'flood'
        
        # Heat stress
        if temp > 35:
            scenarios['heat'] = {
                'type': 'heat',
                'title': '🔥 Heat Stress Conditions',
                'description': f'Temperature ({temp}°C) is very high.',
                'advice': [
                    'Apply mulch to reduce soil temperature',
                    'Increase irrigation frequency',
                    'Use shade netting for sensitive crops',
                    'Plant heat-tolerant varieties',
                    'Avoid midday irrigation (causes leaf burn)'
                ],
                'alert_level': 'error'
            }
        
        # Cold conditions
        if temp < 15:
            cold_crops = ['apple', 'grapes', 'lentil']
            alternatives = [crop for crop, conf in all_predictions if crop in cold_crops][:3]
            
            scenarios['cold'] = {
                'type': 'cold',
                'title': '❄️ Cool/Cold Climate',
                'description': f'Temperature ({temp}°C) is low. Cool-season crops recommended.',
                'alternative_crops': alternatives,
                'advice': [
                    'Delay planting until soil warms',
                    'Use frost protection (covers, greenhouses)',
                    'Select cold-hardy varieties',
                    'Avoid planting tropical crops',
                    'Monitor for frost warnings'
                ],
                'alert_level': 'info'
            }
        
        # Arid conditions
        if rainfall < 80 and humidity < 50:
            scenarios['arid'] = {
                'type': 'arid',
                'title': '🏜️ Arid Climate Conditions',
                'description': 'Low rainfall and humidity. Desert/semi-arid agriculture.',
                'advice': [
                    'Focus on drought-tolerant crops',
                    'Implement water conservation techniques',
                    'Use drip irrigation systems',
                    'Apply heavy mulching (15cm+)',
                    'Consider rainwater harvesting'
                ],
                'alert_level': 'info'
            }
        
        # Tropical conditions
        if rainfall > 200 and temp > 25 and humidity > 70:
            tropical_crops = ['rice', 'banana', 'coconut', 'papaya']
            alternatives = [crop for crop, conf in all_predictions if crop in tropical_crops][:3]
            
            scenarios['tropical'] = {
                'type': 'tropical',
                'title': '🌴 Tropical Climate Ideal',
                'description': 'High rainfall, temperature, and humidity. Perfect for tropical crops.',
                'alternative_crops': alternatives,
                'advice': [
                    'Excellent conditions for high-value tropical crops',
                    'Monitor for fungal diseases (high moisture)',
                    'Ensure proper drainage despite high rain',
                    'Consider perennial tropical crops',
                    'Maximize productivity with intensive cultivation'
                ],
                'alert_level': 'success'
            }
        
        return scenarios

# """
# Make predictions using trained crop recommendation model
# WITH ENHANCEMENTS: Explainability + Edge Case Warnings + Scenario Analysis
# """

# import joblib
# import numpy as np
# import pandas as pd


# class CropPredictor:
#     """
#     A class to handle preprocessing, crop prediction, and generating
#     confidence metrics based on soil and environmental parameters.
#     """

#     def __init__(self):
#         """Initialize model, scaler, and label encoder."""
#         try:
#             self.model = joblib.load("models/crop_model_svm.pkl")
#             self.scaler = joblib.load("models/scaler.pkl")
#             self.label_encoder = joblib.load("models/label_encoder.pkl")

#             print("✅ Model, Scaler, and Encoder loaded successfully.")
#         except Exception as e:
#             print(f"❌ Error loading model components: {e}")
#             self.model, self.scaler, self.label_encoder = None, None, None

#     # -----------------------------------------------------------
#     def preprocess_inputs(self, input_data: dict) -> np.ndarray:
#         """
#         Convert dictionary input into scaled NumPy array for prediction.
#         """
#         try: 
#             df = pd.DataFrame([input_data])
#             X_scaled = self.scaler.transform(df)
#             return X_scaled
#         except Exception as e:
#             print(f"❌ Error in preprocessing: {e}")
#             raise

#     # -----------------------------------------------------------
#     def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
#         """
#         Predict the most suitable crop and return detailed result metrics.
#         """
#         if self.model is None:
#             print("❌ Model not loaded.")
#             return None

#         try:
#             # ✅ Build properly shaped DataFrame (1 row × 7 features)
#             input_df = pd.DataFrame([[
#                 N, P, K, temperature, humidity, ph, rainfall
#             ]], columns=self.feature_names)

#             print(f"Input shape before preprocessing: {input_df.shape}")

#             # ✅ Scale input
#             X_input = self.preprocess_inputs(input_df)
#             print(f"Input shape after preprocessing: {X_input.shape}")

#             # ✅ Prediction
#             prediction = self.model.predict(X_input)
#             crop_name = self.label_encoder.inverse_transform(prediction)[0]

#             # ✅ Confidence estimation
#             if hasattr(self.model, "predict_proba"):
#                 probability = self.model.predict_proba(X_input).max()
#             elif hasattr(self.model, "decision_function"):
#                 score = self.model.decision_function(X_input)
#                 probability = float(1 / (1 + np.exp(-abs(score))))  # sigmoid
#             else:
#                 probability = None

#             # ✅ Basic “cost index” (placeholder for demo)
#             cost_index = np.random.uniform(0.3, 0.9)

#             # ✅ Environmental warnings
#             warnings = []
#             if ph < 5.5:
#                 warnings.append("Soil too acidic — consider adding lime.")
#             elif ph > 7.5:
#                 warnings.append("Soil too alkaline — add compost or organic matter.")

#             if humidity < 30:
#                 warnings.append("Low humidity — irrigation might be required.")
#             elif humidity > 80:
#                 warnings.append("High humidity — monitor for fungal growth.")

#             result = {
#                 "recommended_crop": crop_name,
#                 "confidence": float(probability) if probability is not None else None,
#                 "warnings": warnings,
#                 "cost_index": round(float(cost_index), 2),
#             }

#             print(f"✅ Prediction successful: {result}")
#             return result

#         except Exception as e:
#             print(f"❌ Error making prediction: {e}")
#             return None
    
#     # 
#     # def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
#     #     """
#     #     Generates crop prediction and confidence score.
#     #     """
#     #     if self.model is None:
#     #         print("❌ Model not loaded.")
#     #         return None, None

#     #     try:
#     #         input_data = {
#     #             'N': N,
#     #             'P': P,
#     #             'K': K,
#     #             'temperature': temperature,
#     #             'humidity': humidity,
#     #             'ph': ph,
#     #             'rainfall': rainfall
#     #         }

#     #         # Preprocess and predict
#     #         X_input = self.preprocess_inputs(input_data)

#     #         prediction = self.model.predict(X_input)
#     #         probability = (
#     #             self.model.predict_proba(X_input).max()
#     #             if hasattr(self.model, "predict_proba")
#     #             else None
#     #         )

#     #         print(f"✅ Prediction successful: {prediction[0]}, Confidence: {probability}")
#     #         return prediction[0], probability

#     #     except Exception as e:
#     #         print(f"❌ Error making prediction: {e}")
#     #         return None, None
        
#     #     #----------------------------#
        
        
#     # def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
#     #     """
#     #     Recommend crop based on soil and climate conditions with explanations
        
#     #     Parameters:
#     #     -----------
#     #     N, P, K : float - Nutrient levels
#     #     temperature, humidity, ph, rainfall : float - Climate/soil parameters
        
#     #     Returns:
#     #     --------
#     #     dict : Complete recommendation with explanations
#     #     """
        
#     #         # Prepare input
#     #     input_data = pd.DataFrame({
#     #             'N': [N],
#     #             'P': [P],
#     #             'K': [K],
#     #             'temperature': [temperature],
#     #             'humidity': [humidity],
#     #             'ph': [ph],
#     #             'rainfall': [rainfall]
#     #         })
            
#     #     input_params = {
#     #         'N': N, 'P': P, 'K': K,
#     #         'temperature': temperature,
#     #         'humidity': humidity,
#     #         'ph': ph, 'rainfall': rainfall
#     #     }
        
#     #     # Scale input
#     #     input_scaled = self.scaler.transform(input_data)
        
#     #     # Predict
#     #     prediction = self.model.predict(input_scaled)
#     #     probabilities = self.model.predict_proba(input_scaled)
        
#     #     # Get crop name
#     #     crop_name = self.label_encoder.inverse_transform(prediction)[0]
#     #     confidence = probabilities.max() * 100
        
#     #     # Get top 3 recommendations
#     #     top_3_idx = probabilities[0].argsort()[-3:][::-1]
#     #     top_3_crops = self.label_encoder.inverse_transform(top_3_idx)
#     #     top_3_probs = probabilities[0][top_3_idx] * 100
        
#     #             # Get explanations
#     #     explanations = self.explain_prediction(input_params, crop_name)

#     #     # Check for edge cases
#     #     warnings = self.check_edge_cases(input_params)

#     #     # Get scenario adjustments (NEW)
#     #     scenarios = self.get_scenario_adjustments(input_params)

#     #     result = {
#     #         'recommended_crop': crop_name,
#     #         'confidence': confidence,
#     #         'top_3_recommendations': list(zip(top_3_crops, top_3_probs)),
#     #         'input_parameters': input_params,
#     #         'explanations': explanations,
#     #         'warnings': warnings,
#     #         'scenarios': scenarios  # NEW
#     #     }

#     #     return result
    
#     def explain_prediction(self, input_params, crop_name):
#         """
#         Explain why a crop was recommended based on input parameters
        
#         Parameters:
#         -----------
#         input_params : dict - User's input values
#         crop_name : str - Recommended crop
        
#         Returns:
#         --------
#         list : Explanation strings for each parameter
#         """
#         # Load crop requirements
#         try:
#             crop_req = pd.read_csv('data/processed/crop_requirements_summary.csv')
#             ideal = crop_req[crop_req['label'] == crop_name]
            
#             if ideal.empty:
#                 return ["No ideal requirements data available for this crop."]
            
#             ideal = ideal.iloc[0]
#         except Exception as e:
#             return [f"Unable to load crop requirements data: {e}"]
        
#         explanations = []
        
#         # Check Nitrogen
#         N_val = input_params['N']
#         N_min, N_max, N_avg = ideal['N_min'], ideal['N_max'], ideal['N_avg']
#         if N_min <= N_val <= N_max:
#             diff_pct = abs(N_val - N_avg) / N_avg * 100 if N_avg > 0 else 0
#             if diff_pct < 15:
#                 explanations.append(f"✅ **Nitrogen (N)**: Your value ({N_val} kg/ha) is optimal (ideal: {N_avg:.1f} kg/ha)")
#             else:
#                 explanations.append(f"✓ **Nitrogen (N)**: Your value ({N_val} kg/ha) is acceptable (range: {N_min:.1f}-{N_max:.1f} kg/ha)")
#         else:
#             explanations.append(f"⚠️ **Nitrogen (N)**: Your value ({N_val} kg/ha) is outside optimal range ({N_min:.1f}-{N_max:.1f} kg/ha)")
        
#         # Check Phosphorus
#         P_val = input_params['P']
#         P_min, P_max, P_avg = ideal['P_min'], ideal['P_max'], ideal['P_avg']
#         if P_min <= P_val <= P_max:
#             diff_pct = abs(P_val - P_avg) / P_avg * 100 if P_avg > 0 else 0
#             if diff_pct < 15:
#                 explanations.append(f"✅ **Phosphorus (P)**: Your value ({P_val} kg/ha) is optimal (ideal: {P_avg:.1f} kg/ha)")
#             else:
#                 explanations.append(f"✓ **Phosphorus (P)**: Your value ({P_val} kg/ha) is acceptable (range: {P_min:.1f}-{P_max:.1f} kg/ha)")
#         else:
#             explanations.append(f"⚠️ **Phosphorus (P)**: Your value ({P_val} kg/ha) is outside optimal range ({P_min:.1f}-{P_max:.1f} kg/ha)")
        
#         # Check Potassium
#         K_val = input_params['K']
#         K_min, K_max, K_avg = ideal['K_min'], ideal['K_max'], ideal['K_avg']
#         if K_min <= K_val <= K_max:
#             diff_pct = abs(K_val - K_avg) / K_avg * 100 if K_avg > 0 else 0
#             if diff_pct < 15:
#                 explanations.append(f"✅ **Potassium (K)**: Your value ({K_val} kg/ha) is optimal (ideal: {K_avg:.1f} kg/ha)")
#             else:
#                 explanations.append(f"✓ **Potassium (K)**: Your value ({K_val} kg/ha) is acceptable (range: {K_min:.1f}-{K_max:.1f} kg/ha)")
#         else:
#             explanations.append(f"⚠️ **Potassium (K)**: Your value ({K_val} kg/ha) is outside optimal range ({K_min:.1f}-{K_max:.1f} kg/ha)")
        
#         # Check Temperature
#         temp_val = input_params['temperature']
#         temp_min, temp_max, temp_avg = ideal['temp_min'], ideal['temp_max'], ideal['temp_avg']
#         if temp_min <= temp_val <= temp_max:
#             explanations.append(f"✅ **Temperature**: Your value ({temp_val}°C) is suitable (ideal: {temp_avg:.1f}°C)")
#         else:
#             explanations.append(f"⚠️ **Temperature**: Your value ({temp_val}°C) is outside optimal range ({temp_min:.1f}-{temp_max:.1f}°C)")
        
#         # Check Humidity
#         hum_val = input_params['humidity']
#         hum_min, hum_max, hum_avg = ideal['humidity_min'], ideal['humidity_max'], ideal['humidity_avg']
#         if hum_min <= hum_val <= hum_max:
#             explanations.append(f"✅ **Humidity**: Your value ({hum_val}%) is suitable (ideal: {hum_avg:.1f}%)")
#         else:
#             explanations.append(f"⚠️ **Humidity**: Your value ({hum_val}%) is outside optimal range ({hum_min:.1f}-{hum_max:.1f}%)")
        
#         # Check pH
#         ph_val = input_params['ph']
#         ph_min, ph_max, ph_avg = ideal['ph_min'], ideal['ph_max'], ideal['ph_avg']
#         if ph_min <= ph_val <= ph_max:
#             explanations.append(f"✅ **Soil pH**: Your value ({ph_val}) is suitable (ideal: {ph_avg:.1f})")
#         else:
#             explanations.append(f"⚠️ **Soil pH**: Your value ({ph_val}) is outside optimal range ({ph_min:.1f}-{ph_max:.1f})")
        
#         # Check Rainfall
#         rain_val = input_params['rainfall']
#         rain_min, rain_max, rain_avg = ideal['rainfall_min'], ideal['rainfall_max'], ideal['rainfall_avg']
#         if rain_min <= rain_val <= rain_max:
#             explanations.append(f"✅ **Rainfall**: Your value ({rain_val} mm) is suitable (ideal: {rain_avg:.1f} mm)")
#         else:
#             explanations.append(f"⚠️ **Rainfall**: Your value ({rain_val} mm) is outside optimal range ({rain_min:.1f}-{rain_max:.1f} mm)")
        
#         return explanations
    
#     def check_edge_cases(self, input_params):
#         """
#         Check for unusual or extreme input values and provide warnings
        
#         Parameters:
#         -----------
#         input_params : dict - User's input values
        
#         Returns:
#         --------
#         list : Warning messages for edge cases
#         """
#         warnings = []
        
#         # Nitrogen checks
#         if input_params['N'] > 120:
#             warnings.append("⚠️ **Very High Nitrogen**: Your nitrogen level is unusually high. Consider soil testing and avoid over-fertilization to prevent environmental damage.")
#         elif input_params['N'] < 20:
#             warnings.append("⚠️ **Very Low Nitrogen**: Your nitrogen level is very low. Soil amendment with organic matter or nitrogen fertilizer is highly recommended before planting.")
        
#         # Phosphorus checks
#         if input_params['P'] > 120:
#             warnings.append("⚠️ **Very High Phosphorus**: Excessive phosphorus detected. This may lead to nutrient imbalances and environmental runoff.")
#         elif input_params['P'] < 10:
#             warnings.append("⚠️ **Very Low Phosphorus**: Phosphorus deficiency detected. Consider adding rock phosphate or compost.")
        
#         # Potassium checks
#         if input_params['K'] > 180:
#             warnings.append("⚠️ **Very High Potassium**: Extremely high potassium levels may interfere with calcium and magnesium uptake.")
#         elif input_params['K'] < 15:
#             warnings.append("⚠️ **Very Low Potassium**: Potassium deficiency detected. Add potash fertilizer or wood ash.")
        
#         # Rainfall checks
#         if input_params['rainfall'] < 40:
#             warnings.append("⚠️ **Very Low Rainfall**: Drought conditions detected. Irrigation will be necessary for most crops. Consider drought-tolerant varieties.")
#         elif input_params['rainfall'] > 250:
#             warnings.append("⚠️ **Very High Rainfall**: Excessive rainfall detected. Ensure proper drainage to prevent waterlogging and root rot.")
        
#         # Temperature checks
#         if input_params['temperature'] > 38:
#             warnings.append("⚠️ **Very High Temperature**: Extreme heat detected. Most crops will experience heat stress. Consider shade netting or heat-tolerant varieties.")
#         elif input_params['temperature'] < 12:
#             warnings.append("⚠️ **Very Low Temperature**: Cool temperatures detected. Many tropical crops may not survive. Consider cold-tolerant varieties.")
        
#         # pH checks
#         if input_params['ph'] < 5.0:
#             warnings.append("⚠️ **Acidic Soil**: Your soil is highly acidic. Apply agricultural lime to raise pH for most crops.")
#         elif input_params['ph'] > 8.0:
#             warnings.append("⚠️ **Alkaline Soil**: Your soil is highly alkaline. Apply sulfur or organic matter to lower pH.")
        
#         # Humidity checks
#         if input_params['humidity'] < 30:
#             warnings.append("⚠️ **Very Low Humidity**: Arid conditions detected. Crops may require frequent watering and mulching.")
#         elif input_params['humidity'] > 95:
#             warnings.append("⚠️ **Very High Humidity**: Excessive humidity may promote fungal diseases. Ensure good air circulation.")
        
#         return warnings
    
#     #------enhanced for scenario alerts
#     def get_scenario_adjustments(self, input_params):
        
#         """
#         Provide scenario-based recommendations for extreme conditions
        
#         Parameters:
#         -----------
#         input_params : dict - User's input values
        
#         Returns:
#         --------
#         dict : Scenario information and recommendations
#         """
#         scenarios = {}
        
#         # DROUGHT SCENARIO (Low Rainfall)
#         if input_params['rainfall'] < 80:
#             scenarios['type'] = 'drought'
#             scenarios['severity'] = 'severe' if input_params['rainfall'] < 50 else 'moderate'
#             scenarios['title'] = '🌵 Low Rainfall Detected'
#             scenarios['description'] = f"Your rainfall ({input_params['rainfall']}mm) indicates drought conditions"
#             scenarios['recommendation'] = 'Consider drought-tolerant crops that require minimal water'
#             scenarios['alternative_crops'] = [
#                 'chickpea', 'mothbeans', 'kidneybeans', 'pigeonpeas', 'mungbean'
#             ]
#             scenarios['advice'] = [
#                 '💧 Implement drip irrigation to maximize water efficiency',
#                 '🌾 Apply mulch to retain soil moisture and reduce evaporation',
#                 '🚰 Consider rainwater harvesting for supplemental irrigation',
#                 '📅 Plant early to utilize residual soil moisture',
#                 '🌱 Choose drought-resistant crop varieties'
#             ]
#             scenarios['alert_level'] = 'warning'
        
#         # EXCESS RAIN SCENARIO (High Rainfall)
#         elif input_params['rainfall'] > 250:
#             scenarios['type'] = 'flood'
#             scenarios['severity'] = 'severe' if input_params['rainfall'] > 280 else 'moderate'
#             scenarios['title'] = '🌊 High Rainfall Expected'
#             scenarios['description'] = f"Your rainfall ({input_params['rainfall']}mm) indicates monsoon/flood conditions"
#             scenarios['recommendation'] = 'Ensure proper drainage; select water-tolerant crops'
#             scenarios['alternative_crops'] = [
#                 'rice', 'jute', 'coconut', 'papaya', 'banana'
#             ]
#             scenarios['advice'] = [
#                 '🚜 Ensure proper field drainage to prevent waterlogging',
#                 '🏔️ Consider raised bed cultivation to protect roots',
#                 '🍄 Monitor for fungal diseases due to high moisture',
#                 '⏰ Time planting to avoid peak rainfall periods',
#                 '💪 Select flood-tolerant varieties'
#             ]
#             scenarios['alert_level'] = 'warning'
        
#         # HEAT STRESS SCENARIO (High Temperature)
#         if input_params['temperature'] > 35:
#             heat_scenario = {}
#             heat_scenario['type'] = 'heat'
#             heat_scenario['severity'] = 'severe' if input_params['temperature'] > 40 else 'moderate'
#             heat_scenario['title'] = '🌡️ High Temperature - Heat Stress Risk'
#             heat_scenario['description'] = f"Temperature ({input_params['temperature']}°C) may cause heat stress in many crops"
#             heat_scenario['recommendation'] = 'Choose heat-tolerant varieties and provide shade if possible'
#             heat_scenario['alternative_crops'] = [
#                 'cotton', 'pigeonpeas', 'mungbean', 'groundnuts', 'watermelon'
#             ]
#             heat_scenario['advice'] = [
#                 '☂️ Install shade nets to reduce direct sun exposure (30-50% shade)',
#                 '💦 Increase irrigation frequency to combat heat stress',
#                 '🌅 Schedule irrigation during early morning or evening',
#                 '🌾 Apply reflective mulch to reduce soil temperature',
#                 '🌡️ Plant heat-tolerant crop varieties'
#             ]
#             heat_scenario['alert_level'] = 'error'
#             scenarios['heat'] = heat_scenario
        
#         # COLD SCENARIO (Low Temperature)
#         elif input_params['temperature'] < 15:
#             cold_scenario = {}
#             cold_scenario['type'] = 'cold'
#             cold_scenario['severity'] = 'severe' if input_params['temperature'] < 10 else 'moderate'
#             cold_scenario['title'] = '❄️ Cool Temperature - Cold Sensitive Crops at Risk'
#             cold_scenario['description'] = f"Temperature ({input_params['temperature']}°C) too low for tropical crops"
#             cold_scenario['recommendation'] = 'Select cold-tolerant varieties; protect sensitive crops'
#             cold_scenario['alternative_crops'] = [
#                 'apple', 'lentil', 'chickpea', 'grapes', 'pomegranate'
#             ]
#             cold_scenario['advice'] = [
#                 '🛡️ Use row covers or plastic tunnels for frost protection',
#                 '🔥 Consider windbreaks to reduce cold wind damage',
#                 '📅 Delay planting until soil warms up',
#                 '❄️ Avoid tropical/subtropical crops (banana, papaya, coconut)',
#                 '🌱 Plant cold-hardy varieties'
#             ]
#             cold_scenario['alert_level'] = 'info'
#             scenarios['cold'] = cold_scenario
        
#         # ARID SCENARIO (Low Rainfall + High Temperature + Low Humidity)
#         if (input_params['rainfall'] < 80 and 
#             input_params['temperature'] > 30 and 
#             input_params['humidity'] < 40):
            
#             arid_scenario = {}
#             arid_scenario['type'] = 'arid'
#             arid_scenario['severity'] = 'severe'
#             arid_scenario['title'] = '🏜️ Arid/Desert Conditions Detected'
#             arid_scenario['description'] = 'Combination of low rainfall, high heat, and low humidity'
#             arid_scenario['recommendation'] = 'Only drought-resistant crops will survive without irrigation'
#             arid_scenario['alternative_crops'] = [
#                 'mothbeans', 'chickpea', 'pigeonpeas', 'groundnuts'
#             ]
#             arid_scenario['advice'] = [
#                 '💧 Drip irrigation is essential for most crops',
#                 '🌵 Consider xerophytic (desert-adapted) crop varieties',
#                 '🌾 Deep mulching (10-15cm) to conserve moisture',
#                 '🌱 Reduce plant density to minimize water competition',
#                 '🚰 Invest in water storage infrastructure'
#             ]
#             arid_scenario['alert_level'] = 'error'
#             scenarios['arid'] = arid_scenario
        
#         # TROPICAL/MONSOON SCENARIO (High Rainfall + High Temperature + High Humidity)
#         if (input_params['rainfall'] > 200 and 
#             input_params['temperature'] > 28 and 
#             input_params['humidity'] > 80):
            
#             tropical_scenario = {}
#             tropical_scenario['type'] = 'tropical'
#             tropical_scenario['severity'] = 'moderate'
#             tropical_scenario['title'] = '🌴 Tropical/Monsoon Climate Detected'
#             tropical_scenario['description'] = 'Ideal for tropical crops but monitor for diseases'
#             tropical_scenario['recommendation'] = 'Excellent for rice, coconut, and tropical fruits'
#             tropical_scenario['alternative_crops'] = [
#                 'rice', 'coconut', 'banana', 'papaya', 'mango'
#             ]
#             tropical_scenario['advice'] = [
#                 '✅ Ideal conditions for tropical crop cultivation',
#                 '🍄 Monitor closely for fungal diseases (high humidity)',
#                 '🌾 Ensure good air circulation between plants',
#                 '💚 Take advantage of year-round growing season',
#                 '🌴 Consider high-value tropical fruits for better returns'
#             ]
#             tropical_scenario['alert_level'] = 'success'
#             scenarios['tropical'] = tropical_scenario
        
#         return scenarios  
#         #----------------enhanced
    
    
#     def display_prediction(self, result):
#         """Display prediction results in formatted way"""
#         print()
#         print("🌾 CROP RECOMMENDATION RESULT")
#         print()
        
#         print(f"\n✅ RECOMMENDED CROP: {result['recommended_crop'].upper()}")
#         print(f"   Confidence: {result['confidence']:.2f}%")
        
#         if result['warnings']:
#             print(f"\n⚠️ WARNINGS:")
#             for warning in result['warnings']:
#                 print(f"   {warning}")
        
#         print(f"\n🔍 EXPLANATION:")
#         for explanation in result['explanations']:
#             print(f"   {explanation}")
        
#         print(f"\n📊 TOP 3 RECOMMENDATIONS:")
#         for i, (crop, prob) in enumerate(result['top_3_recommendations'], 1):
#             print(f"   {i}. {crop:<15} - {prob:.2f}%")
        
#         print(f"\n📋 INPUT PARAMETERS:")
#         params = result['input_parameters']
#         print(f"   Nitrogen (N):    {params['N']} kg/ha")
#         print(f"   Phosphorus (P):  {params['P']} kg/ha")
#         print(f"   Potassium (K):   {params['K']} kg/ha")
#         print(f"   Temperature:     {params['temperature']}°C")
#         print(f"   Humidity:        {params['humidity']}%")
#         print(f"   pH:              {params['ph']}")
#         print(f"   Rainfall:        {params['rainfall']} mm")
        
#         print()


# # Example usage
# if __name__ == "__main__":
#     # Initialize predictor
#     predictor = CropPredictor()
    
#     # # Example 1: Rice-suitable conditions
#     # print("Example 1: Rice-suitable conditions")
#     # result1 = predictor.recommend_crop(
#     #     N=90, P=42, K=43,
#     #     temperature=21, humidity=82,
#     #     ph=6.5, rainfall=202
#     # )
#     # predictor.display_prediction(result1)
    
#     # # Example 2: Coffee-suitable conditions
#     # print("\nExample 2: Coffee-suitable conditions")
#     # result2 = predictor.recommend_crop(
#     #     N=101, P=32, K=30,
#     #     temperature=23, humidity=58,
#     #     ph=6.8, rainfall=140
#     # )
#     # predictor.display_prediction(result2)
    
#     # # Example 3: Extreme conditions (with warnings)
#     # print("\nExample 3: Extreme conditions")
#     # result3 = predictor.recommend_crop(
#     #     N=150, P=10, K=200,
#     #     temperature=40, humidity=20,
#     #     ph=4.0, rainfall=30
#     # )
#     # predictor.display_prediction(result3)

# # """
# # Make predictions using trained crop recommendation model
# # """

# # import joblib
# # import pandas as pd
# # import numpy as np

# # class CropPredictor:
# #     """Class to handle crop predictions"""
    
# #     def __init__(self, model_path='models/crop_model_random_forest.pkl',
# #                  scaler_path='models/scaler.pkl',
# #                  encoder_path='models/label_encoder.pkl'):
# #         """Load trained model and preprocessing objects"""
# #         self.model = joblib.load(model_path)
# #         self.scaler = joblib.load(scaler_path)
# #         self.label_encoder = joblib.load(encoder_path)
# #         print("✅ Model loaded successfully")
    
# # # Replace the existing recommend_crop method with this enhanced version

# # #-------------------------------------------------------
# # def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
# #     """
# #     Recommend crop based on soil and climate conditions with explanations
# #     """
# #     # Prepare input
# #     input_data = pd.DataFrame({
# #         'N': [N],
# #         'P': [P],
# #         'K': [K],
# #         'temperature': [temperature],
# #         'humidity': [humidity],
# #         'ph': [ph],
# #         'rainfall': [rainfall]
# #     })
    
# #     input_params = {
# #         'N': N, 'P': P, 'K': K,
# #         'temperature': temperature,
# #         'humidity': humidity,
# #         'ph': ph, 'rainfall': rainfall
# #     }
    
# #     # Scale input
# #     input_scaled = self.scaler.transform(input_data)
    
# #     # Predict
# #     prediction = self.model.predict(input_scaled)
# #     probabilities = self.model.predict_proba(input_scaled)
    
# #     # Get crop name
# #     crop_name = self.encoder.inverse_transform(prediction)[0]
# #     confidence = probabilities.max() * 100
    
# #     # Get top 3 recommendations
# #     top_3_idx = probabilities[0].argsort()[-3:][::-1]
# #     top_3_crops = self.encoder.inverse_transform(top_3_idx)
# #     top_3_probs = probabilities[0][top_3_idx] * 100
    
# #     # Get explanations
# #     explanations = self.explain_prediction(input_params, crop_name)
    
# #     # Check for edge cases
# #     warnings = self.check_edge_cases(input_params)
    
# #     result = {
# #         'recommended_crop': crop_name,
# #         'confidence': confidence,
# #         'top_3_recommendations': list(zip(top_3_crops, top_3_probs)),
# #         'input_parameters': input_params,
# #         'explanations': explanations,
# #         'warnings': warnings
# #     }
    
# #     return result
    
# #     def predict_batch(self, input_file, output_file=None):
# #         """
# #         Predict crops for batch of samples from CSV file
        
# #         Parameters:
# #         -----------
# #         input_file : str - Path to input CSV file
# #         output_file : str - Path to save predictions (optional)
        
# #         Returns:
# #         --------
# #         DataFrame : Predictions for all samples
# #         """
# #         # Load input data
# #         df = pd.read_csv(input_file)
        
# #         # Extract features
# #         feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
# #         X = df[feature_cols]
        
# #         # Scale
# #         X_scaled = self.scaler.transform(X)
        
# #         # Predict
# #         predictions = self.model.predict(X_scaled)
# #         probabilities = self.model.predict_proba(X_scaled)
        
# #         # Convert to crop names
# #         crop_names = self.label_encoder.inverse_transform(predictions)
# #         confidences = probabilities.max(axis=1) * 100
        
# #         # Add predictions to dataframe
# #         df['predicted_crop'] = crop_names
# #         df['confidence'] = confidences.round(2)
        
# #         # Save if output file specified
# #         if output_file:
# #             df.to_csv(output_file, index=False)
# #             print(f"✅ Predictions saved to: {output_file}")
        
# #         return df
    
# #     def display_prediction(self, result):
# #         """Display prediction results in formatted way"""
# #         print("\n" + "="*60)
# #         print("🌾 CROP RECOMMENDATION RESULT")
# #         print("="*60)
        
# #         print(f"\n✅ RECOMMENDED CROP: {result['recommended_crop'].upper()}")
# #         print(f"   Confidence: {result['confidence']:.2f}%")
        
# #         print(f"\n📊 TOP 3 RECOMMENDATIONS:")
# #         for i, rec in enumerate(result['top_3_recommendations'], 1):
# #             print(f"   {i}. {rec['crop']:<15} - {rec['confidence']:.2f}%")
        
# #         print(f"\n📋 INPUT PARAMETERS:")
# #         params = result['input_parameters']
# #         print(f"   Nitrogen (N):    {params['N']} kg/ha")
# #         print(f"   Phosphorus (P):  {params['P']} kg/ha")
# #         print(f"   Potassium (K):   {params['K']} kg/ha")
# #         print(f"   Temperature:     {params['temperature']}°C")
# #         print(f"   Humidity:        {params['humidity']}%")
# #         print(f"   pH:              {params['ph']}")
# #         print(f"   Rainfall:        {params['rainfall']} mm")
        
# #         print("="*60 + "\n")

# #     # Add this method to CropPredictor class (around line 50)

# # def explain_prediction(self, input_params, crop_name):
# #     """
# #     Explain why a crop was recommended based on input parameters
    
# #     Parameters:
# #     -----------
# #     input_params : dict - User's input values
# #     crop_name : str - Recommended crop
    
# #     Returns:
# #     --------
# #     list : Explanation strings for each parameter
# #     """
# #     # Load crop requirements
# #     try:
# #         crop_req = pd.read_csv('data/processed/crop_requirements_summary.csv')
# #         ideal = crop_req[crop_req['crop'] == crop_name]
        
# #         if ideal.empty:
# #             return ["No ideal requirements data available for this crop."]
        
# #         ideal = ideal.iloc[0]
# #     except:
# #         return ["Unable to load crop requirements data."]
    
# #     explanations = []
    
# #     # Check Nitrogen
# #     N_val = input_params['N']
# #     N_min, N_max, N_avg = ideal['N_min'], ideal['N_max'], ideal['N_avg']
# #     if N_min <= N_val <= N_max:
# #         diff_pct = abs(N_val - N_avg) / N_avg * 100
# #         if diff_pct < 15:
# #             explanations.append(f"✅ **Nitrogen (N)**: Your value ({N_val} kg/ha) is optimal (ideal: {N_avg:.1f} kg/ha)")
# #         else:
# #             explanations.append(f"✓ **Nitrogen (N)**: Your value ({N_val} kg/ha) is acceptable (range: {N_min:.1f}-{N_max:.1f} kg/ha)")
# #     else:
# #         explanations.append(f"⚠️ **Nitrogen (N)**: Your value ({N_val} kg/ha) is outside optimal range ({N_min:.1f}-{N_max:.1f} kg/ha)")
    
# #     # Check Phosphorus
# #     P_val = input_params['P']
# #     P_min, P_max, P_avg = ideal['P_min'], ideal['P_max'], ideal['P_avg']
# #     if P_min <= P_val <= P_max:
# #         diff_pct = abs(P_val - P_avg) / P_avg * 100
# #         if diff_pct < 15:
# #             explanations.append(f"✅ **Phosphorus (P)**: Your value ({P_val} kg/ha) is optimal (ideal: {P_avg:.1f} kg/ha)")
# #         else:
# #             explanations.append(f"✓ **Phosphorus (P)**: Your value ({P_val} kg/ha) is acceptable (range: {P_min:.1f}-{P_max:.1f} kg/ha)")
# #     else:
# #         explanations.append(f"⚠️ **Phosphorus (P)**: Your value ({P_val} kg/ha) is outside optimal range ({P_min:.1f}-{P_max:.1f} kg/ha)")
    
# #     # Check Potassium
# #     K_val = input_params['K']
# #     K_min, K_max, K_avg = ideal['K_min'], ideal['K_max'], ideal['K_avg']
# #     if K_min <= K_val <= K_max:
# #         diff_pct = abs(K_val - K_avg) / K_avg * 100
# #         if diff_pct < 15:
# #             explanations.append(f"✅ **Potassium (K)**: Your value ({K_val} kg/ha) is optimal (ideal: {K_avg:.1f} kg/ha)")
# #         else:
# #             explanations.append(f"✓ **Potassium (K)**: Your value ({K_val} kg/ha) is acceptable (range: {K_min:.1f}-{K_max:.1f} kg/ha)")
# #     else:
# #         explanations.append(f"⚠️ **Potassium (K)**: Your value ({K_val} kg/ha) is outside optimal range ({K_min:.1f}-{K_max:.1f} kg/ha)")
    
# #     # Check Temperature
# #     temp_val = input_params['temperature']
# #     temp_min, temp_max, temp_avg = ideal['temp_min'], ideal['temp_max'], ideal['temp_avg']
# #     if temp_min <= temp_val <= temp_max:
# #         explanations.append(f"✅ **Temperature**: Your value ({temp_val}°C) is suitable (ideal: {temp_avg:.1f}°C)")
# #     else:
# #         explanations.append(f"⚠️ **Temperature**: Your value ({temp_val}°C) is outside optimal range ({temp_min:.1f}-{temp_max:.1f}°C)")
    
# #     # Check Humidity
# #     hum_val = input_params['humidity']
# #     hum_min, hum_max, hum_avg = ideal['humidity_min'], ideal['humidity_max'], ideal['humidity_avg']
# #     if hum_min <= hum_val <= hum_max:
# #         explanations.append(f"✅ **Humidity**: Your value ({hum_val}%) is suitable (ideal: {hum_avg:.1f}%)")
# #     else:
# #         explanations.append(f"⚠️ **Humidity**: Your value ({hum_val}%) is outside optimal range ({hum_min:.1f}-{hum_max:.1f}%)")
    
# #     # Check pH
# #     ph_val = input_params['ph']
# #     ph_min, ph_max, ph_avg = ideal['ph_min'], ideal['ph_max'], ideal['ph_avg']
# #     if ph_min <= ph_val <= ph_max:
# #         explanations.append(f"✅ **Soil pH**: Your value ({ph_val}) is suitable (ideal: {ph_avg:.1f})")
# #     else:
# #         explanations.append(f"⚠️ **Soil pH**: Your value ({ph_val}) is outside optimal range ({ph_min:.1f}-{ph_max:.1f})")
    
# #     # Check Rainfall
# #     rain_val = input_params['rainfall']
# #     rain_min, rain_max, rain_avg = ideal['rainfall_min'], ideal['rainfall_max'], ideal['rainfall_avg']
# #     if rain_min <= rain_val <= rain_max:
# #         explanations.append(f"✅ **Rainfall**: Your value ({rain_val} mm) is suitable (ideal: {rain_avg:.1f} mm)")
# #     else:
# #         explanations.append(f"⚠️ **Rainfall**: Your value ({rain_val} mm) is outside optimal range ({rain_min:.1f}-{rain_max:.1f} mm)")
    
# #     return explanations

# # def check_edge_cases(self, input_params):
# #     """
# #     Check for unusual or extreme input values and provide warnings
    
# #     Parameters:
# #     -----------
# #     input_params : dict - User's input values
    
# #     Returns:
# #     --------
# #     list : Warning messages for edge cases
# #     """
# #     warnings = []
    
# #     # Nitrogen checks
# #     if input_params['N'] > 120:
# #         warnings.append("⚠️ **Very High Nitrogen**: Your nitrogen level is unusually high. Consider soil testing and avoid over-fertilization to prevent environmental damage.")
# #     elif input_params['N'] < 20:
# #         warnings.append("⚠️ **Very Low Nitrogen**: Your nitrogen level is very low. Soil amendment with organic matter or nitrogen fertilizer is highly recommended before planting.")
    
# #     # Phosphorus checks
# #     if input_params['P'] > 120:
# #         warnings.append("⚠️ **Very High Phosphorus**: Excessive phosphorus detected. This may lead to nutrient imbalances and environmental runoff.")
# #     elif input_params['P'] < 10:
# #         warnings.append("⚠️ **Very Low Phosphorus**: Phosphorus deficiency detected. Consider adding rock phosphate or compost.")
    
# #     # Potassium checks
# #     if input_params['K'] > 180:
# #         warnings.append("⚠️ **Very High Potassium**: Extremely high potassium levels may interfere with calcium and magnesium uptake.")
# #     elif input_params['K'] < 15:
# #         warnings.append("⚠️ **Very Low Potassium**: Potassium deficiency detected. Add potash fertilizer or wood ash.")
    
# #     # Rainfall checks
# #     if input_params['rainfall'] < 40:
# #         warnings.append("⚠️ **Very Low Rainfall**: Drought conditions detected. Irrigation will be necessary for most crops. Consider drought-tolerant varieties.")
# #     elif input_params['rainfall'] > 250:
# #         warnings.append("⚠️ **Very High Rainfall**: Excessive rainfall detected. Ensure proper drainage to prevent waterlogging and root rot.")
    
# #     # Temperature checks
# #     if input_params['temperature'] > 38:
# #         warnings.append("⚠️ **Very High Temperature**: Extreme heat detected. Most crops will experience heat stress. Consider shade netting or heat-tolerant varieties.")
# #     elif input_params['temperature'] < 12:
# #         warnings.append("⚠️ **Very Low Temperature**: Cool temperatures detected. Many tropical crops may not survive. Consider cold-tolerant varieties.")
    
# #     # pH checks
# #     if input_params['ph'] < 5.0:
# #         warnings.append("⚠️ **Acidic Soil**: Your soil is highly acidic. Apply agricultural lime to raise pH for most crops.")
# #     elif input_params['ph'] > 8.0:
# #         warnings.append("⚠️ **Alkaline Soil**: Your soil is highly alkaline. Apply sulfur or organic matter to lower pH.")
    
# #     # Humidity checks
# #     if input_params['humidity'] < 30:
# #         warnings.append("⚠️ **Very Low Humidity**: Arid conditions detected. Crops may require frequent watering and mulching.")
# #     elif input_params['humidity'] > 95:
# #         warnings.append("⚠️ **Very High Humidity**: Excessive humidity may promote fungal diseases. Ensure good air circulation.")
    
# #     return warnings
    
# #     #------------------------------------------------------
    
# # # usage
# # if __name__ == "__main__":
# #     # Initialize predictor
# #     predictor = CropPredictor()
    
# #     # # Example 1: Single prediction
# #     # print("Example 1: Rice-suitable conditions")
# #     # result1 = predictor.predict_single(
# #     #     N=90, P=42, K=43,
# #     #     temperature=21, humidity=82,
# #     #     ph=6.5, rainfall=202
# #     # )
# #     # predictor.display_prediction(result1)
    
# #     # # Example 2: Coffee-suitable conditions
# #     # print("\nExample 2: Coffee-suitable conditions")
# #     # result2 = predictor.predict_single(
# #     #     N=101, P=32, K=30,
# #     #     temperature=23, humidity=58,
# #     #     ph=6.8, rainfall=140
# #     # )
# #     # predictor.display_prediction(result2)
    
# #     # # Example 3: Watermelon-suitable conditions
# #     # print("\nExample 3: Watermelon-suitable conditions")
# #     # result3 = predictor.predict_single(
# #     #     N=100, P=80, K=120,
# #     #     temperature=27, humidity=84,
# #     #     ph=7.0, rainfall=60
# #     # )
# #     # predictor.display_prediction(result3)