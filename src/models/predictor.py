import pandas as pd
import joblib
import numpy as np

class CropPredictor:
    def __init__(self, model_path):
        """Initialize predictor and safely load the model."""
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None

    # ---------------------------------------------------------
    # 1️⃣ Preprocessing
    # ---------------------------------------------------------
    def preprocess_inputs(self, input_data):
        """
        Recreate engineered features and align columns with training schema.
        """
        df = pd.DataFrame([input_data])

        # --- Derived numeric ratios ---
        df['NP_ratio'] = df['N'] / (df['P'] + 1e-6)
        df['NK_ratio'] = df['N'] / (df['K'] + 1e-6)
        df['NPK_ratio'] = (df['N'] + df['P'] + df['K']) / 3

        # --- Nutrient categories (same bins as during training) ---
        df['N_category'] = pd.cut(df['N'], bins=[0, 50, 100, 150],
                                  labels=['Low', 'Medium', 'High'], include_lowest=True)
        df['K_category'] = pd.cut(df['K'], bins=[0, 50, 100, 200],
                                  labels=['Low', 'Medium', 'High'], include_lowest=True)

        # --- One-hot encoding for categorical features ---
        df = pd.get_dummies(df, drop_first=False)

        # --- Column alignment with model training features ---
        if hasattr(self.model, "feature_names_in_"):
            df = df.reindex(columns=self.model.feature_names_in_, fill_value=0)
        else:
            print("⚠️ Model missing feature_names_in_. Ensure consistent input order.")
        return df

    # ---------------------------------------------------------
    # 2️⃣ Crop Recommendation
    # ---------------------------------------------------------
    def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
        """Make a prediction and return crop + confidence."""
        if self.model is None:
            print("❌ Model not loaded.")
            return None, None

        try:
                # Ensure consistent feature order with training
            input_data = pd.DataFrame([{
                    'N': N, 
                    'P': P, 
                    'K': K,
                    'temperature': temperature,
                    'humidity': humidity,
                    'ph': ph,
                    'rainfall': rainfall
                }])

                # Preprocess safely
            X_input = self.preprocess_inputs(input_data)

                # Ensure column names persist after preprocessing
            if isinstance(X_input, np.ndarray):
                X_input = pd.DataFrame(X_input, columns=self.feature_names)

            prediction = self.model.predict(X_input)[0]

                # Confidence score
            probability = None
            if hasattr(self.model, "predict_proba"):
                probability = float(self.model.predict_proba(X_input).max())
            elif hasattr(self.model, "decision_function"):
                probability = float(1 / (1 + np.exp(-abs(self.model.decision_function(X_input)))))

            print(f"✅ Prediction: {prediction}, Confidence: {probability}")
            return prediction, probability

        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None, None


    # ---------------------------------------------------------
    # 3️⃣ Explainability
    # ---------------------------------------------------------
    def explain_prediction(self, input_data):
        """Explain key contributing parameters."""
        explanation = {}
        for feature, value in input_data.items():
            if feature in ['N', 'P', 'K']:
                if value < 40:
                    explanation[feature] = "Low nutrient level — may limit growth."
                elif value > 100:
                    explanation[feature] = "High nutrient — suitable for heavy feeders."
                else:
                    explanation[feature] = "Optimal range."
            elif feature == 'rainfall':
                if value < 50:
                    explanation[feature] = "Low rainfall — drought-tolerant crops preferred."
                elif value > 150:
                    explanation[feature] = "High rainfall — water-intensive crops preferred."
                else:
                    explanation[feature] = "Moderate rainfall — versatile conditions."
            elif feature == 'ph':
                if value < 5.5:
                    explanation[feature] = "Soil acidic — liming may be required."
                elif value > 8:
                    explanation[feature] = "Alkaline soil — crop options limited."
                else:
                    explanation[feature] = "Ideal soil pH."
            elif feature == 'temperature':
                if value < 18:
                    explanation[feature] = "Cool climate — temperate crops suitable."
                elif value > 35:
                    explanation[feature] = "Hot climate — drought-tolerant crops suitable."
                else:
                    explanation[feature] = "Favorable temperature range."
        return explanation

    # ---------------------------------------------------------
    # 4️⃣ Edge Case Detection
    # ---------------------------------------------------------
    def check_extreme_conditions(self, input_data):
        """Identify unrealistic or extreme input values."""
        warnings = []
        if input_data['N'] > 150 or input_data['K'] > 200:
            warnings.append("⚠️ Nutrient values abnormally high. Verify measurements.")
        if input_data['ph'] < 3.5 or input_data['ph'] > 9.5:
            warnings.append("⚠️ Unusual pH — likely input error.")
        if input_data['temperature'] < 5 or input_data['temperature'] > 45:
            warnings.append("⚠️ Extreme temperature range — check data accuracy.")
        if input_data['rainfall'] < 10 or input_data['rainfall'] > 300:
            warnings.append("⚠️ Rainfall extreme — model reliability may drop.")
        return warnings

    # ---------------------------------------------------------
    # 5️⃣ Climate Scenario Simulation
    # ---------------------------------------------------------
    def simulate_climate_change(self, input_data, delta_temp=2.0, delta_rain=-10):
        """Simulate effect of climate changes."""
        adjusted_data = input_data.copy()
        adjusted_data['temperature'] += delta_temp
        adjusted_data['rainfall'] = max(0, adjusted_data['rainfall'] + delta_rain)

        crop, conf = self.recommend_crop(**adjusted_data)
        return adjusted_data, crop, conf
