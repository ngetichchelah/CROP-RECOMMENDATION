"""
Make predictions using trained crop recommendation model
WITH ENHANCEMENTS: Explainability + Edge Case Warnings + Scenario Analysis
"""

import joblib
import pandas as pd
import numpy as np

class CropPredictor:
    """Class to handle crop predictions with explainability"""
    
    def __init__(self, 
                 model_path='models/crop_model_svm.pkl',
                 scaler_path='models/scaler.pkl',
                 encoder_path='models/label_encoder.pkl'):
        """Load trained model and preprocessing objects"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def explain_prediction(self, input_params, crop_name):
        """
        Explain why a crop was recommended based on input parameters
        
        Parameters:
        -----------
        input_params : dict - User's input values
        crop_name : str - Recommended crop
        
        Returns:
        --------
        list : Explanation strings for each parameter
        """
        # Load crop requirements
        try:
            crop_req = pd.read_csv('data/processed/crop_requirements_summary.csv')
            ideal = crop_req[crop_req['label'] == crop_name]
            
            if ideal.empty:
                return ["No ideal requirements data available for this crop."]
            
            ideal = ideal.iloc[0]
        except Exception as e:
            return [f"Unable to load crop requirements data: {e}"]
        
        explanations = []
        
        # Check Nitrogen
        N_val = input_params['N']
        N_min, N_max, N_avg = ideal['N_min'], ideal['N_max'], ideal['N_avg']
        if N_min <= N_val <= N_max:
            diff_pct = abs(N_val - N_avg) / N_avg * 100 if N_avg > 0 else 0
            if diff_pct < 15:
                explanations.append(f"✅ **Nitrogen (N)**: Your value ({N_val} kg/ha) is optimal (ideal: {N_avg:.1f} kg/ha)")
            else:
                explanations.append(f"✓ **Nitrogen (N)**: Your value ({N_val} kg/ha) is acceptable (range: {N_min:.1f}-{N_max:.1f} kg/ha)")
        else:
            explanations.append(f"⚠️ **Nitrogen (N)**: Your value ({N_val} kg/ha) is outside optimal range ({N_min:.1f}-{N_max:.1f} kg/ha)")
        
        # Check Phosphorus
        P_val = input_params['P']
        P_min, P_max, P_avg = ideal['P_min'], ideal['P_max'], ideal['P_avg']
        if P_min <= P_val <= P_max:
            diff_pct = abs(P_val - P_avg) / P_avg * 100 if P_avg > 0 else 0
            if diff_pct < 15:
                explanations.append(f"✅ **Phosphorus (P)**: Your value ({P_val} kg/ha) is optimal (ideal: {P_avg:.1f} kg/ha)")
            else:
                explanations.append(f"✓ **Phosphorus (P)**: Your value ({P_val} kg/ha) is acceptable (range: {P_min:.1f}-{P_max:.1f} kg/ha)")
        else:
            explanations.append(f"⚠️ **Phosphorus (P)**: Your value ({P_val} kg/ha) is outside optimal range ({P_min:.1f}-{P_max:.1f} kg/ha)")
        
        # Check Potassium
        K_val = input_params['K']
        K_min, K_max, K_avg = ideal['K_min'], ideal['K_max'], ideal['K_avg']
        if K_min <= K_val <= K_max:
            diff_pct = abs(K_val - K_avg) / K_avg * 100 if K_avg > 0 else 0
            if diff_pct < 15:
                explanations.append(f"✅ **Potassium (K)**: Your value ({K_val} kg/ha) is optimal (ideal: {K_avg:.1f} kg/ha)")
            else:
                explanations.append(f"✓ **Potassium (K)**: Your value ({K_val} kg/ha) is acceptable (range: {K_min:.1f}-{K_max:.1f} kg/ha)")
        else:
            explanations.append(f"⚠️ **Potassium (K)**: Your value ({K_val} kg/ha) is outside optimal range ({K_min:.1f}-{K_max:.1f} kg/ha)")
        
        # Check Temperature
        temp_val = input_params['temperature']
        temp_min, temp_max, temp_avg = ideal['temp_min'], ideal['temp_max'], ideal['temp_avg']
        if temp_min <= temp_val <= temp_max:
            explanations.append(f"✅ **Temperature**: Your value ({temp_val}°C) is suitable (ideal: {temp_avg:.1f}°C)")
        else:
            explanations.append(f"⚠️ **Temperature**: Your value ({temp_val}°C) is outside optimal range ({temp_min:.1f}-{temp_max:.1f}°C)")
        
        # Check Humidity
        hum_val = input_params['humidity']
        hum_min, hum_max, hum_avg = ideal['humidity_min'], ideal['humidity_max'], ideal['humidity_avg']
        if hum_min <= hum_val <= hum_max:
            explanations.append(f"✅ **Humidity**: Your value ({hum_val}%) is suitable (ideal: {hum_avg:.1f}%)")
        else:
            explanations.append(f"⚠️ **Humidity**: Your value ({hum_val}%) is outside optimal range ({hum_min:.1f}-{hum_max:.1f}%)")
        
        # Check pH
        ph_val = input_params['ph']
        ph_min, ph_max, ph_avg = ideal['ph_min'], ideal['ph_max'], ideal['ph_avg']
        if ph_min <= ph_val <= ph_max:
            explanations.append(f"✅ **Soil pH**: Your value ({ph_val}) is suitable (ideal: {ph_avg:.1f})")
        else:
            explanations.append(f"⚠️ **Soil pH**: Your value ({ph_val}) is outside optimal range ({ph_min:.1f}-{ph_max:.1f})")
        
        # Check Rainfall
        rain_val = input_params['rainfall']
        rain_min, rain_max, rain_avg = ideal['rainfall_min'], ideal['rainfall_max'], ideal['rainfall_avg']
        if rain_min <= rain_val <= rain_max:
            explanations.append(f"✅ **Rainfall**: Your value ({rain_val} mm) is suitable (ideal: {rain_avg:.1f} mm)")
        else:
            explanations.append(f"⚠️ **Rainfall**: Your value ({rain_val} mm) is outside optimal range ({rain_min:.1f}-{rain_max:.1f} mm)")
        
        return explanations
    
    def check_edge_cases(self, input_params):
        """
        Check for unusual or extreme input values and provide warnings
        
        Parameters:
        -----------
        input_params : dict - User's input values
        
        Returns:
        --------
        list : Warning messages for edge cases
        """
        warnings = []
        
        # Nitrogen checks
        if input_params['N'] > 120:
            warnings.append("⚠️ **Very High Nitrogen**: Your nitrogen level is unusually high. Consider soil testing and avoid over-fertilization to prevent environmental damage.")
        elif input_params['N'] < 20:
            warnings.append("⚠️ **Very Low Nitrogen**: Your nitrogen level is very low. Soil amendment with organic matter or nitrogen fertilizer is highly recommended before planting.")
        
        # Phosphorus checks
        if input_params['P'] > 120:
            warnings.append("⚠️ **Very High Phosphorus**: Excessive phosphorus detected. This may lead to nutrient imbalances and environmental runoff.")
        elif input_params['P'] < 10:
            warnings.append("⚠️ **Very Low Phosphorus**: Phosphorus deficiency detected. Consider adding rock phosphate or compost.")
        
        # Potassium checks
        if input_params['K'] > 180:
            warnings.append("⚠️ **Very High Potassium**: Extremely high potassium levels may interfere with calcium and magnesium uptake.")
        elif input_params['K'] < 15:
            warnings.append("⚠️ **Very Low Potassium**: Potassium deficiency detected. Add potash fertilizer or wood ash.")
        
        # Rainfall checks
        if input_params['rainfall'] < 40:
            warnings.append("⚠️ **Very Low Rainfall**: Drought conditions detected. Irrigation will be necessary for most crops. Consider drought-tolerant varieties.")
        elif input_params['rainfall'] > 250:
            warnings.append("⚠️ **Very High Rainfall**: Excessive rainfall detected. Ensure proper drainage to prevent waterlogging and root rot.")
        
        # Temperature checks
        if input_params['temperature'] > 38:
            warnings.append("⚠️ **Very High Temperature**: Extreme heat detected. Most crops will experience heat stress. Consider shade netting or heat-tolerant varieties.")
        elif input_params['temperature'] < 12:
            warnings.append("⚠️ **Very Low Temperature**: Cool temperatures detected. Many tropical crops may not survive. Consider cold-tolerant varieties.")
        
        # pH checks
        if input_params['ph'] < 5.0:
            warnings.append("⚠️ **Acidic Soil**: Your soil is highly acidic. Apply agricultural lime to raise pH for most crops.")
        elif input_params['ph'] > 8.0:
            warnings.append("⚠️ **Alkaline Soil**: Your soil is highly alkaline. Apply sulfur or organic matter to lower pH.")
        
        # Humidity checks
        if input_params['humidity'] < 30:
            warnings.append("⚠️ **Very Low Humidity**: Arid conditions detected. Crops may require frequent watering and mulching.")
        elif input_params['humidity'] > 95:
            warnings.append("⚠️ **Very High Humidity**: Excessive humidity may promote fungal diseases. Ensure good air circulation.")
        
        return warnings
    
    #------enhanced for scenario alerts
    def get_scenario_adjustments(self, input_params):
        
        """
        Provide scenario-based recommendations for extreme conditions
        
        Parameters:
        -----------
        input_params : dict - User's input values
        
        Returns:
        --------
        dict : Scenario information and recommendations
        """
        scenarios = {}
        
        # DROUGHT SCENARIO (Low Rainfall)
        if input_params['rainfall'] < 80:
            scenarios['type'] = 'drought'
            scenarios['severity'] = 'severe' if input_params['rainfall'] < 50 else 'moderate'
            scenarios['title'] = '🌵 Low Rainfall Detected'
            scenarios['description'] = f"Your rainfall ({input_params['rainfall']}mm) indicates drought conditions"
            scenarios['recommendation'] = 'Consider drought-tolerant crops that require minimal water'
            scenarios['alternative_crops'] = [
                'chickpea', 'mothbeans', 'kidneybeans', 'pigeonpeas', 'mungbean'
            ]
            scenarios['advice'] = [
                '💧 Implement drip irrigation to maximize water efficiency',
                '🌾 Apply mulch to retain soil moisture and reduce evaporation',
                '🚰 Consider rainwater harvesting for supplemental irrigation',
                '📅 Plant early to utilize residual soil moisture',
                '🌱 Choose drought-resistant crop varieties'
            ]
            scenarios['alert_level'] = 'warning'
        
        # EXCESS RAIN SCENARIO (High Rainfall)
        elif input_params['rainfall'] > 250:
            scenarios['type'] = 'flood'
            scenarios['severity'] = 'severe' if input_params['rainfall'] > 280 else 'moderate'
            scenarios['title'] = '🌊 High Rainfall Expected'
            scenarios['description'] = f"Your rainfall ({input_params['rainfall']}mm) indicates monsoon/flood conditions"
            scenarios['recommendation'] = 'Ensure proper drainage; select water-tolerant crops'
            scenarios['alternative_crops'] = [
                'rice', 'jute', 'coconut', 'papaya', 'banana'
            ]
            scenarios['advice'] = [
                '🚜 Ensure proper field drainage to prevent waterlogging',
                '🏔️ Consider raised bed cultivation to protect roots',
                '🍄 Monitor for fungal diseases due to high moisture',
                '⏰ Time planting to avoid peak rainfall periods',
                '💪 Select flood-tolerant varieties'
            ]
            scenarios['alert_level'] = 'warning'
        
        # HEAT STRESS SCENARIO (High Temperature)
        if input_params['temperature'] > 35:
            heat_scenario = {}
            heat_scenario['type'] = 'heat'
            heat_scenario['severity'] = 'severe' if input_params['temperature'] > 40 else 'moderate'
            heat_scenario['title'] = '🌡️ High Temperature - Heat Stress Risk'
            heat_scenario['description'] = f"Temperature ({input_params['temperature']}°C) may cause heat stress in many crops"
            heat_scenario['recommendation'] = 'Choose heat-tolerant varieties and provide shade if possible'
            heat_scenario['alternative_crops'] = [
                'cotton', 'pigeonpeas', 'mungbean', 'groundnuts', 'watermelon'
            ]
            heat_scenario['advice'] = [
                '☂️ Install shade nets to reduce direct sun exposure (30-50% shade)',
                '💦 Increase irrigation frequency to combat heat stress',
                '🌅 Schedule irrigation during early morning or evening',
                '🌾 Apply reflective mulch to reduce soil temperature',
                '🌡️ Plant heat-tolerant crop varieties'
            ]
            heat_scenario['alert_level'] = 'error'
            scenarios['heat'] = heat_scenario
        
        # COLD SCENARIO (Low Temperature)
        elif input_params['temperature'] < 15:
            cold_scenario = {}
            cold_scenario['type'] = 'cold'
            cold_scenario['severity'] = 'severe' if input_params['temperature'] < 10 else 'moderate'
            cold_scenario['title'] = '❄️ Cool Temperature - Cold Sensitive Crops at Risk'
            cold_scenario['description'] = f"Temperature ({input_params['temperature']}°C) too low for tropical crops"
            cold_scenario['recommendation'] = 'Select cold-tolerant varieties; protect sensitive crops'
            cold_scenario['alternative_crops'] = [
                'apple', 'lentil', 'chickpea', 'grapes', 'pomegranate'
            ]
            cold_scenario['advice'] = [
                '🛡️ Use row covers or plastic tunnels for frost protection',
                '🔥 Consider windbreaks to reduce cold wind damage',
                '📅 Delay planting until soil warms up',
                '❄️ Avoid tropical/subtropical crops (banana, papaya, coconut)',
                '🌱 Plant cold-hardy varieties'
            ]
            cold_scenario['alert_level'] = 'info'
            scenarios['cold'] = cold_scenario
        
        # ARID SCENARIO (Low Rainfall + High Temperature + Low Humidity)
        if (input_params['rainfall'] < 80 and 
            input_params['temperature'] > 30 and 
            input_params['humidity'] < 40):
            
            arid_scenario = {}
            arid_scenario['type'] = 'arid'
            arid_scenario['severity'] = 'severe'
            arid_scenario['title'] = '🏜️ Arid/Desert Conditions Detected'
            arid_scenario['description'] = 'Combination of low rainfall, high heat, and low humidity'
            arid_scenario['recommendation'] = 'Only drought-resistant crops will survive without irrigation'
            arid_scenario['alternative_crops'] = [
                'mothbeans', 'chickpea', 'pigeonpeas', 'groundnuts'
            ]
            arid_scenario['advice'] = [
                '💧 Drip irrigation is essential for most crops',
                '🌵 Consider xerophytic (desert-adapted) crop varieties',
                '🌾 Deep mulching (10-15cm) to conserve moisture',
                '🌱 Reduce plant density to minimize water competition',
                '🚰 Invest in water storage infrastructure'
            ]
            arid_scenario['alert_level'] = 'error'
            scenarios['arid'] = arid_scenario
        
        # TROPICAL/MONSOON SCENARIO (High Rainfall + High Temperature + High Humidity)
        if (input_params['rainfall'] > 200 and 
            input_params['temperature'] > 28 and 
            input_params['humidity'] > 80):
            
            tropical_scenario = {}
            tropical_scenario['type'] = 'tropical'
            tropical_scenario['severity'] = 'moderate'
            tropical_scenario['title'] = '🌴 Tropical/Monsoon Climate Detected'
            tropical_scenario['description'] = 'Ideal for tropical crops but monitor for diseases'
            tropical_scenario['recommendation'] = 'Excellent for rice, coconut, and tropical fruits'
            tropical_scenario['alternative_crops'] = [
                'rice', 'coconut', 'banana', 'papaya', 'mango'
            ]
            tropical_scenario['advice'] = [
                '✅ Ideal conditions for tropical crop cultivation',
                '🍄 Monitor closely for fungal diseases (high humidity)',
                '🌾 Ensure good air circulation between plants',
                '💚 Take advantage of year-round growing season',
                '🌴 Consider high-value tropical fruits for better returns'
            ]
            tropical_scenario['alert_level'] = 'success'
            scenarios['tropical'] = tropical_scenario
        
        return scenarios  
        #----------------enhanced
    
    def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Recommend crop based on soil and climate conditions with explanations
        
        Parameters:
        -----------
        N, P, K : float - Nutrient levels
        temperature, humidity, ph, rainfall : float - Climate/soil parameters
        
        Returns:
        --------
        dict : Complete recommendation with explanations
        """
        # Prepare input
        input_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })
        
        input_params = {
            'N': N, 'P': P, 'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph, 'rainfall': rainfall
        }
        
        # Scale input
        input_scaled = self.scaler.transform(input_data)
        
        # Predict
        prediction = self.model.predict(input_scaled)
        probabilities = self.model.predict_proba(input_scaled)
        
        # Get crop name
        crop_name = self.label_encoder.inverse_transform(prediction)[0]
        confidence = probabilities.max() * 100
        
        # Get top 3 recommendations
        top_3_idx = probabilities[0].argsort()[-3:][::-1]
        top_3_crops = self.label_encoder.inverse_transform(top_3_idx)
        top_3_probs = probabilities[0][top_3_idx] * 100
        
                # Get explanations
        explanations = self.explain_prediction(input_params, crop_name)

        # Check for edge cases
        warnings = self.check_edge_cases(input_params)

        # Get scenario adjustments (NEW)
        scenarios = self.get_scenario_adjustments(input_params)

        result = {
            'recommended_crop': crop_name,
            'confidence': confidence,
            'top_3_recommendations': list(zip(top_3_crops, top_3_probs)),
            'input_parameters': input_params,
            'explanations': explanations,
            'warnings': warnings,
            'scenarios': scenarios  # NEW
        }

        return result
    
    def display_prediction(self, result):
        """Display prediction results in formatted way"""
        print()
        print("🌾 CROP RECOMMENDATION RESULT")
        print()
        
        print(f"\n✅ RECOMMENDED CROP: {result['recommended_crop'].upper()}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        
        if result['warnings']:
            print(f"\n⚠️ WARNINGS:")
            for warning in result['warnings']:
                print(f"   {warning}")
        
        print(f"\n🔍 EXPLANATION:")
        for explanation in result['explanations']:
            print(f"   {explanation}")
        
        print(f"\n📊 TOP 3 RECOMMENDATIONS:")
        for i, (crop, prob) in enumerate(result['top_3_recommendations'], 1):
            print(f"   {i}. {crop:<15} - {prob:.2f}%")
        
        print(f"\n📋 INPUT PARAMETERS:")
        params = result['input_parameters']
        print(f"   Nitrogen (N):    {params['N']} kg/ha")
        print(f"   Phosphorus (P):  {params['P']} kg/ha")
        print(f"   Potassium (K):   {params['K']} kg/ha")
        print(f"   Temperature:     {params['temperature']}°C")
        print(f"   Humidity:        {params['humidity']}%")
        print(f"   pH:              {params['ph']}")
        print(f"   Rainfall:        {params['rainfall']} mm")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = CropPredictor()
    
    # # Example 1: Rice-suitable conditions
    # print("Example 1: Rice-suitable conditions")
    # result1 = predictor.recommend_crop(
    #     N=90, P=42, K=43,
    #     temperature=21, humidity=82,
    #     ph=6.5, rainfall=202
    # )
    # predictor.display_prediction(result1)
    
    # # Example 2: Coffee-suitable conditions
    # print("\nExample 2: Coffee-suitable conditions")
    # result2 = predictor.recommend_crop(
    #     N=101, P=32, K=30,
    #     temperature=23, humidity=58,
    #     ph=6.8, rainfall=140
    # )
    # predictor.display_prediction(result2)
    
    # # Example 3: Extreme conditions (with warnings)
    # print("\nExample 3: Extreme conditions")
    # result3 = predictor.recommend_crop(
    #     N=150, P=10, K=200,
    #     temperature=40, humidity=20,
    #     ph=4.0, rainfall=30
    # )
    # predictor.display_prediction(result3)

# """
# Make predictions using trained crop recommendation model
# """

# import joblib
# import pandas as pd
# import numpy as np

# class CropPredictor:
#     """Class to handle crop predictions"""
    
#     def __init__(self, model_path='models/crop_model_random_forest.pkl',
#                  scaler_path='models/scaler.pkl',
#                  encoder_path='models/label_encoder.pkl'):
#         """Load trained model and preprocessing objects"""
#         self.model = joblib.load(model_path)
#         self.scaler = joblib.load(scaler_path)
#         self.label_encoder = joblib.load(encoder_path)
#         print("✅ Model loaded successfully")
    
# # Replace the existing recommend_crop method with this enhanced version

# #-------------------------------------------------------
# def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
#     """
#     Recommend crop based on soil and climate conditions with explanations
#     """
#     # Prepare input
#     input_data = pd.DataFrame({
#         'N': [N],
#         'P': [P],
#         'K': [K],
#         'temperature': [temperature],
#         'humidity': [humidity],
#         'ph': [ph],
#         'rainfall': [rainfall]
#     })
    
#     input_params = {
#         'N': N, 'P': P, 'K': K,
#         'temperature': temperature,
#         'humidity': humidity,
#         'ph': ph, 'rainfall': rainfall
#     }
    
#     # Scale input
#     input_scaled = self.scaler.transform(input_data)
    
#     # Predict
#     prediction = self.model.predict(input_scaled)
#     probabilities = self.model.predict_proba(input_scaled)
    
#     # Get crop name
#     crop_name = self.encoder.inverse_transform(prediction)[0]
#     confidence = probabilities.max() * 100
    
#     # Get top 3 recommendations
#     top_3_idx = probabilities[0].argsort()[-3:][::-1]
#     top_3_crops = self.encoder.inverse_transform(top_3_idx)
#     top_3_probs = probabilities[0][top_3_idx] * 100
    
#     # Get explanations
#     explanations = self.explain_prediction(input_params, crop_name)
    
#     # Check for edge cases
#     warnings = self.check_edge_cases(input_params)
    
#     result = {
#         'recommended_crop': crop_name,
#         'confidence': confidence,
#         'top_3_recommendations': list(zip(top_3_crops, top_3_probs)),
#         'input_parameters': input_params,
#         'explanations': explanations,
#         'warnings': warnings
#     }
    
#     return result
    
#     def predict_batch(self, input_file, output_file=None):
#         """
#         Predict crops for batch of samples from CSV file
        
#         Parameters:
#         -----------
#         input_file : str - Path to input CSV file
#         output_file : str - Path to save predictions (optional)
        
#         Returns:
#         --------
#         DataFrame : Predictions for all samples
#         """
#         # Load input data
#         df = pd.read_csv(input_file)
        
#         # Extract features
#         feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
#         X = df[feature_cols]
        
#         # Scale
#         X_scaled = self.scaler.transform(X)
        
#         # Predict
#         predictions = self.model.predict(X_scaled)
#         probabilities = self.model.predict_proba(X_scaled)
        
#         # Convert to crop names
#         crop_names = self.label_encoder.inverse_transform(predictions)
#         confidences = probabilities.max(axis=1) * 100
        
#         # Add predictions to dataframe
#         df['predicted_crop'] = crop_names
#         df['confidence'] = confidences.round(2)
        
#         # Save if output file specified
#         if output_file:
#             df.to_csv(output_file, index=False)
#             print(f"✅ Predictions saved to: {output_file}")
        
#         return df
    
#     def display_prediction(self, result):
#         """Display prediction results in formatted way"""
#         print("\n" + "="*60)
#         print("🌾 CROP RECOMMENDATION RESULT")
#         print("="*60)
        
#         print(f"\n✅ RECOMMENDED CROP: {result['recommended_crop'].upper()}")
#         print(f"   Confidence: {result['confidence']:.2f}%")
        
#         print(f"\n📊 TOP 3 RECOMMENDATIONS:")
#         for i, rec in enumerate(result['top_3_recommendations'], 1):
#             print(f"   {i}. {rec['crop']:<15} - {rec['confidence']:.2f}%")
        
#         print(f"\n📋 INPUT PARAMETERS:")
#         params = result['input_parameters']
#         print(f"   Nitrogen (N):    {params['N']} kg/ha")
#         print(f"   Phosphorus (P):  {params['P']} kg/ha")
#         print(f"   Potassium (K):   {params['K']} kg/ha")
#         print(f"   Temperature:     {params['temperature']}°C")
#         print(f"   Humidity:        {params['humidity']}%")
#         print(f"   pH:              {params['ph']}")
#         print(f"   Rainfall:        {params['rainfall']} mm")
        
#         print("="*60 + "\n")

#     # Add this method to CropPredictor class (around line 50)

# def explain_prediction(self, input_params, crop_name):
#     """
#     Explain why a crop was recommended based on input parameters
    
#     Parameters:
#     -----------
#     input_params : dict - User's input values
#     crop_name : str - Recommended crop
    
#     Returns:
#     --------
#     list : Explanation strings for each parameter
#     """
#     # Load crop requirements
#     try:
#         crop_req = pd.read_csv('data/processed/crop_requirements_summary.csv')
#         ideal = crop_req[crop_req['crop'] == crop_name]
        
#         if ideal.empty:
#             return ["No ideal requirements data available for this crop."]
        
#         ideal = ideal.iloc[0]
#     except:
#         return ["Unable to load crop requirements data."]
    
#     explanations = []
    
#     # Check Nitrogen
#     N_val = input_params['N']
#     N_min, N_max, N_avg = ideal['N_min'], ideal['N_max'], ideal['N_avg']
#     if N_min <= N_val <= N_max:
#         diff_pct = abs(N_val - N_avg) / N_avg * 100
#         if diff_pct < 15:
#             explanations.append(f"✅ **Nitrogen (N)**: Your value ({N_val} kg/ha) is optimal (ideal: {N_avg:.1f} kg/ha)")
#         else:
#             explanations.append(f"✓ **Nitrogen (N)**: Your value ({N_val} kg/ha) is acceptable (range: {N_min:.1f}-{N_max:.1f} kg/ha)")
#     else:
#         explanations.append(f"⚠️ **Nitrogen (N)**: Your value ({N_val} kg/ha) is outside optimal range ({N_min:.1f}-{N_max:.1f} kg/ha)")
    
#     # Check Phosphorus
#     P_val = input_params['P']
#     P_min, P_max, P_avg = ideal['P_min'], ideal['P_max'], ideal['P_avg']
#     if P_min <= P_val <= P_max:
#         diff_pct = abs(P_val - P_avg) / P_avg * 100
#         if diff_pct < 15:
#             explanations.append(f"✅ **Phosphorus (P)**: Your value ({P_val} kg/ha) is optimal (ideal: {P_avg:.1f} kg/ha)")
#         else:
#             explanations.append(f"✓ **Phosphorus (P)**: Your value ({P_val} kg/ha) is acceptable (range: {P_min:.1f}-{P_max:.1f} kg/ha)")
#     else:
#         explanations.append(f"⚠️ **Phosphorus (P)**: Your value ({P_val} kg/ha) is outside optimal range ({P_min:.1f}-{P_max:.1f} kg/ha)")
    
#     # Check Potassium
#     K_val = input_params['K']
#     K_min, K_max, K_avg = ideal['K_min'], ideal['K_max'], ideal['K_avg']
#     if K_min <= K_val <= K_max:
#         diff_pct = abs(K_val - K_avg) / K_avg * 100
#         if diff_pct < 15:
#             explanations.append(f"✅ **Potassium (K)**: Your value ({K_val} kg/ha) is optimal (ideal: {K_avg:.1f} kg/ha)")
#         else:
#             explanations.append(f"✓ **Potassium (K)**: Your value ({K_val} kg/ha) is acceptable (range: {K_min:.1f}-{K_max:.1f} kg/ha)")
#     else:
#         explanations.append(f"⚠️ **Potassium (K)**: Your value ({K_val} kg/ha) is outside optimal range ({K_min:.1f}-{K_max:.1f} kg/ha)")
    
#     # Check Temperature
#     temp_val = input_params['temperature']
#     temp_min, temp_max, temp_avg = ideal['temp_min'], ideal['temp_max'], ideal['temp_avg']
#     if temp_min <= temp_val <= temp_max:
#         explanations.append(f"✅ **Temperature**: Your value ({temp_val}°C) is suitable (ideal: {temp_avg:.1f}°C)")
#     else:
#         explanations.append(f"⚠️ **Temperature**: Your value ({temp_val}°C) is outside optimal range ({temp_min:.1f}-{temp_max:.1f}°C)")
    
#     # Check Humidity
#     hum_val = input_params['humidity']
#     hum_min, hum_max, hum_avg = ideal['humidity_min'], ideal['humidity_max'], ideal['humidity_avg']
#     if hum_min <= hum_val <= hum_max:
#         explanations.append(f"✅ **Humidity**: Your value ({hum_val}%) is suitable (ideal: {hum_avg:.1f}%)")
#     else:
#         explanations.append(f"⚠️ **Humidity**: Your value ({hum_val}%) is outside optimal range ({hum_min:.1f}-{hum_max:.1f}%)")
    
#     # Check pH
#     ph_val = input_params['ph']
#     ph_min, ph_max, ph_avg = ideal['ph_min'], ideal['ph_max'], ideal['ph_avg']
#     if ph_min <= ph_val <= ph_max:
#         explanations.append(f"✅ **Soil pH**: Your value ({ph_val}) is suitable (ideal: {ph_avg:.1f})")
#     else:
#         explanations.append(f"⚠️ **Soil pH**: Your value ({ph_val}) is outside optimal range ({ph_min:.1f}-{ph_max:.1f})")
    
#     # Check Rainfall
#     rain_val = input_params['rainfall']
#     rain_min, rain_max, rain_avg = ideal['rainfall_min'], ideal['rainfall_max'], ideal['rainfall_avg']
#     if rain_min <= rain_val <= rain_max:
#         explanations.append(f"✅ **Rainfall**: Your value ({rain_val} mm) is suitable (ideal: {rain_avg:.1f} mm)")
#     else:
#         explanations.append(f"⚠️ **Rainfall**: Your value ({rain_val} mm) is outside optimal range ({rain_min:.1f}-{rain_max:.1f} mm)")
    
#     return explanations

# def check_edge_cases(self, input_params):
#     """
#     Check for unusual or extreme input values and provide warnings
    
#     Parameters:
#     -----------
#     input_params : dict - User's input values
    
#     Returns:
#     --------
#     list : Warning messages for edge cases
#     """
#     warnings = []
    
#     # Nitrogen checks
#     if input_params['N'] > 120:
#         warnings.append("⚠️ **Very High Nitrogen**: Your nitrogen level is unusually high. Consider soil testing and avoid over-fertilization to prevent environmental damage.")
#     elif input_params['N'] < 20:
#         warnings.append("⚠️ **Very Low Nitrogen**: Your nitrogen level is very low. Soil amendment with organic matter or nitrogen fertilizer is highly recommended before planting.")
    
#     # Phosphorus checks
#     if input_params['P'] > 120:
#         warnings.append("⚠️ **Very High Phosphorus**: Excessive phosphorus detected. This may lead to nutrient imbalances and environmental runoff.")
#     elif input_params['P'] < 10:
#         warnings.append("⚠️ **Very Low Phosphorus**: Phosphorus deficiency detected. Consider adding rock phosphate or compost.")
    
#     # Potassium checks
#     if input_params['K'] > 180:
#         warnings.append("⚠️ **Very High Potassium**: Extremely high potassium levels may interfere with calcium and magnesium uptake.")
#     elif input_params['K'] < 15:
#         warnings.append("⚠️ **Very Low Potassium**: Potassium deficiency detected. Add potash fertilizer or wood ash.")
    
#     # Rainfall checks
#     if input_params['rainfall'] < 40:
#         warnings.append("⚠️ **Very Low Rainfall**: Drought conditions detected. Irrigation will be necessary for most crops. Consider drought-tolerant varieties.")
#     elif input_params['rainfall'] > 250:
#         warnings.append("⚠️ **Very High Rainfall**: Excessive rainfall detected. Ensure proper drainage to prevent waterlogging and root rot.")
    
#     # Temperature checks
#     if input_params['temperature'] > 38:
#         warnings.append("⚠️ **Very High Temperature**: Extreme heat detected. Most crops will experience heat stress. Consider shade netting or heat-tolerant varieties.")
#     elif input_params['temperature'] < 12:
#         warnings.append("⚠️ **Very Low Temperature**: Cool temperatures detected. Many tropical crops may not survive. Consider cold-tolerant varieties.")
    
#     # pH checks
#     if input_params['ph'] < 5.0:
#         warnings.append("⚠️ **Acidic Soil**: Your soil is highly acidic. Apply agricultural lime to raise pH for most crops.")
#     elif input_params['ph'] > 8.0:
#         warnings.append("⚠️ **Alkaline Soil**: Your soil is highly alkaline. Apply sulfur or organic matter to lower pH.")
    
#     # Humidity checks
#     if input_params['humidity'] < 30:
#         warnings.append("⚠️ **Very Low Humidity**: Arid conditions detected. Crops may require frequent watering and mulching.")
#     elif input_params['humidity'] > 95:
#         warnings.append("⚠️ **Very High Humidity**: Excessive humidity may promote fungal diseases. Ensure good air circulation.")
    
#     return warnings
    
#     #------------------------------------------------------
    
# # usage
# if __name__ == "__main__":
#     # Initialize predictor
#     predictor = CropPredictor()
    
#     # # Example 1: Single prediction
#     # print("Example 1: Rice-suitable conditions")
#     # result1 = predictor.predict_single(
#     #     N=90, P=42, K=43,
#     #     temperature=21, humidity=82,
#     #     ph=6.5, rainfall=202
#     # )
#     # predictor.display_prediction(result1)
    
#     # # Example 2: Coffee-suitable conditions
#     # print("\nExample 2: Coffee-suitable conditions")
#     # result2 = predictor.predict_single(
#     #     N=101, P=32, K=30,
#     #     temperature=23, humidity=58,
#     #     ph=6.8, rainfall=140
#     # )
#     # predictor.display_prediction(result2)
    
#     # # Example 3: Watermelon-suitable conditions
#     # print("\nExample 3: Watermelon-suitable conditions")
#     # result3 = predictor.predict_single(
#     #     N=100, P=80, K=120,
#     #     temperature=27, humidity=84,
#     #     ph=7.0, rainfall=60
#     # )
#     # predictor.display_prediction(result3)