"""
Streamlit Web Application for Crop Recommendation System
WITH ENHANCEMENTS: Explainability + Edge Case Warnings
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f8f0;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
    }
    .crop-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .metric-card {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load CropPredictor
@st.cache_resource
def load_predictor():
    """Load CropPredictor instance"""
    try:
        from src.models.predict import CropPredictor
        predictor = CropPredictor()
        return predictor
    except Exception as e:
        st.error(f"❌ Error loading predictor: {e}")
        st.error("Make sure src/models/predict.py exists with CropPredictor class")
        st.error(f"Current directory: {os.getcwd()}")
        return None

# Load crop information
@st.cache_data
def load_crop_data():
    """Load crop requirements data"""
    try:
        crop_req = pd.read_csv('data/processed/crop_requirements_summary.csv')
        return crop_req
    except Exception as e:
        st.warning(f"Could not load crop requirements: {e}")
        return None

# Main app
def main():
    # Load predictor
    predictor = load_predictor()
    crop_data = load_crop_data()
    
    if predictor is None:
        st.error("❌ Failed to load predictor. Please check:")
        st.error("1. models/ folder exists with crop_model_svm.pkl, scaler.pkl, label_encoder.pkl")
        st.error("2. src/models/predict.py exists with CropPredictor class")
        return
    
    # Header
    st.title("🌾 Crop Recommendation System")
    st.markdown("### Make data-driven decisions for optimal crop selection")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📊 About")
    st.sidebar.info("""
    This intelligent system recommends the best crop to plant based on:
    - Soil nutrients (NPK)
    - Climate conditions
    - Soil pH
    - Rainfall patterns
    
    **Accuracy: 93.24%**
    **Model: Support Vector Machine (SVM)**
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 SDG Impact")
    st.sidebar.success("""
    **SDG 2: Zero Hunger**
    - Increase yields by 15-25%
    - Reduce fertilizer waste by 30%
    - Support sustainable agriculture
    """)
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Soil & Climate Parameters")
        
        with st.form("prediction_form"):
            # Nutrient inputs
            st.subheader("🧪 Soil Nutrients")
            col_n, col_p, col_k = st.columns(3)
            
            with col_n:
                N = st.number_input(
                    "Nitrogen (N) kg/ha",
                    min_value=0,
                    max_value=140,
                    value=90,
                    help="Nitrogen content in soil (0-140 kg/ha)"
                )
            
            with col_p:
                P = st.number_input(
                    "Phosphorus (P) kg/ha",
                    min_value=5,
                    max_value=145,
                    value=42,
                    help="Phosphorus content in soil (5-145 kg/ha)"
                )
            
            with col_k:
                K = st.number_input(
                    "Potassium (K) kg/ha",
                    min_value=5,
                    max_value=205,
                    value=43,
                    help="Potassium content in soil (5-205 kg/ha)"
                )
            
            st.markdown("---")
            
            # Climate inputs
            st.subheader("🌡️ Climate Conditions")
            col_temp, col_hum = st.columns(2)
            
            with col_temp:
                temperature = st.slider(
                    "Temperature (°C)",
                    min_value=8.0,
                    max_value=44.0,
                    value=21.0,
                    step=0.5,
                    help="Average temperature (8-44°C)"
                )
            
            with col_hum:
                humidity = st.slider(
                    "Humidity (%)",
                    min_value=14.0,
                    max_value=100.0,
                    value=82.0,
                    step=1.0,
                    help="Relative humidity (14-100%)"
                )
            
            st.markdown("---")
            
            # Soil pH and Rainfall
            st.subheader("🌧️ Additional Parameters")
            col_ph, col_rain = st.columns(2)
            
            with col_ph:
                ph = st.slider(
                    "Soil pH",
                    min_value=3.5,
                    max_value=9.9,
                    value=6.5,
                    step=0.1,
                    help="Soil pH level (3.5-9.9)"
                )
            
            with col_rain:
                rainfall = st.slider(
                    "Rainfall (mm)",
                    min_value=20.0,
                    max_value=300.0,
                    value=202.0,
                    step=5.0,
                    help="Annual rainfall (20-300mm)"
                )
            
            st.markdown("---")
            
            # Submit button
            submitted = st.form_submit_button("🔍 Get Recommendation", use_container_width=True)
    
    with col2:
        st.header("🎯 Recommendation Results")
        
        if submitted:
            # Make prediction using CropPredictor
            try:
                result = predictor.recommend_crop(
                    N=N, P=P, K=K,
                    temperature=temperature,
                    humidity=humidity,
                    ph=ph,
                    rainfall=rainfall
                )
                
                crop_name = result['recommended_crop']
                confidence = result['confidence']
                top_3_recommendations = result['top_3_recommendations']
                explanations = result['explanations']
                warnings = result['warnings']
                
                # Display warnings first (if any)
                if warnings:
                    st.warning("### ⚠️ Important Alerts")
                    for warning in warnings:
                        st.markdown(warning)
                    st.markdown("---")
                
                # Display main recommendation
                st.markdown(f"""
                <div class="crop-card">
                    <h1 style='text-align: center; color: #2E7D32;'>🌾 {crop_name.upper()}</h1>
                    <h3 style='text-align: center; color: #666;'>Recommended Crop</h3>
                    <div class="metric-card">
                        <h2 style='color: #1B5E20; margin: 0;'>{confidence:.2f}%</h2>
                        <p style='margin: 5px 0 0 0;'>Confidence Score</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar for confidence
                st.progress(confidence / 100)
                
                st.markdown("---")
            #-------------------------------------------(enhanced)
                # Import crop profiles
                try:
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                    from src.utils.crops_profiles import get_crop_profile
                    
                    # Get profile for recommended crop
                    profile = get_crop_profile(crop_name)
                    
                    st.subheader("📊 Crop Profile & Economic Analysis")
                    
                    # Display metrics in 4 columns
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        # Color-code economic value
                        econ_color = {
                            'Very High': '🟢',
                            'High': '🟢', 
                            'Medium': '🟡',
                            'Low': '🔴',
                            'N/A': '⚪'
                        }
                        st.metric(
                            "Economic Value", 
                            f"{econ_color.get(profile['economic_value'], '⚪')} {profile['economic_value']}"
                        )
                    
                    with metric_col2:
                        # Color-code water efficiency
                        water_color = {
                            'Very High': '🟢',
                            'High': '🟢',
                            'Medium': '🟡',
                            'Low': '🔴',
                            'N/A': '⚪'
                        }
                        st.metric(
                            "Water Efficiency", 
                            f"{water_color.get(profile['water_efficiency'], '⚪')} {profile['water_efficiency']}"
                        )
                    
                    with metric_col3:
                        # Color-code market demand
                        demand_color = {
                            'Very High': '🟢',
                            'High': '🟢',
                            'Medium': '🟡',
                            'Low': '🔴',
                            'N/A': '⚪'
                        }
                        st.metric(
                            "Market Demand", 
                            f"{demand_color.get(profile['market_demand'], '⚪')} {profile['market_demand']}"
                        )
                    
                    with metric_col4:
                        # Color-code sustainability
                        sust_color = {
                            'Very High': '🟢',
                            'High': '🟢',
                            'Medium': '🟡',
                            'Low': '🔴',
                            'N/A': '⚪'
                        }
                        st.metric(
                            "Sustainability", 
                            f"{sust_color.get(profile['sustainability'], '⚪')} {profile['sustainability']}"
                        )
                    
                    # Additional economic info in 2 columns
                    econ_col1, econ_col2 = st.columns(2)
                    
                    with econ_col1:
                        st.info(f"**Expected Yield**: {profile['typical_yield']}")
                        st.info(f"**Labor Intensity**: {profile['labor_intensity']}")
                    
                    with econ_col2:
                        st.info(f"**Market Price**: {profile['market_price']}")
                        st.success(f"**💡 Tip**: {profile['description']}")
                    
                except ImportError as e:
                    st.warning(f"Could not load crop profiles: {e}")
                except Exception as e:
                    st.warning(f"Error displaying crop profile: {e}")
                                
                #-----------------------------------            
                
    ##--------------------enhanced- scenario display-------------------
                                
                                # Display scenario analysis (NEW)
                scenarios = result.get('scenarios', {})

                if scenarios:
                    st.markdown("---")
                    st.subheader("🌦️ Climate Scenario Analysis")
                    
                    # Primary scenario (drought/flood)
                    if 'type' in scenarios and scenarios['type'] in ['drought', 'flood']:
                        scenario = scenarios
                        
                        # Choose alert type based on severity
                        if scenario['alert_level'] == 'warning':
                            st.warning(f"### {scenario['title']}")
                        elif scenario['alert_level'] == 'error':
                            st.error(f"### {scenario['title']}")
                        else:
                            st.info(f"### {scenario['title']}")
                        
                        st.write(f"**{scenario['description']}**")
                        st.write(f"**Recommendation**: {scenario['recommendation']}")
                        
                        # Alternative crops
                        if 'alternative_crops' in scenario:
                            st.info(f"**🌾 Alternative crops for this scenario**: {', '.join(scenario['alternative_crops'])}")
                        
                        # Advice
                        if 'advice' in scenario:
                            with st.expander("📋 Detailed Farming Recommendations", expanded=True):
                                for advice_item in scenario['advice']:
                                    st.write(f"• {advice_item}")
                    
                    # Secondary scenarios (heat/cold/arid/tropical)
                    secondary_scenarios = [
                        scenarios.get('heat'),
                        scenarios.get('cold'),
                        scenarios.get('arid'),
                        scenarios.get('tropical')
                    ]
                    
                    for scenario in secondary_scenarios:
                        if scenario:
                            # Choose alert type
                            if scenario['alert_level'] == 'error':
                                st.error(f"**{scenario['title']}**")
                            elif scenario['alert_level'] == 'warning':
                                st.warning(f"**{scenario['title']}**")
                            elif scenario['alert_level'] == 'success':
                                st.success(f"**{scenario['title']}**")
                            else:
                                st.info(f"**{scenario['title']}**")
                            
                            st.write(scenario['description'])
                            
                            if 'alternative_crops' in scenario:
                                st.write(f"**Suitable crops**: {', '.join(scenario['alternative_crops'])}")
                            
                            # Show advice in expander
                            if 'advice' in scenario:
                                with st.expander(f"View recommendations for {scenario['type']} conditions"):
                                    for advice_item in scenario['advice']:
                                        st.write(f"• {advice_item}")
                                
                                
                #####----------------------------end of enhanced scenario display
                
                # Explanation Section
                st.subheader("🔍 Why This Crop?")
                st.info("Here's how your soil and climate conditions match this crop's requirements:")
                
                for explanation in explanations:
                    if "✅" in explanation:
                        st.success(explanation)
                    elif "✓" in explanation:
                        st.info(explanation)
                    else:
                        st.warning(explanation)
                
                st.markdown("---")
                
                # Top 3 recommendations
                st.subheader("📊 Top 3 Alternative Crops")
                
                # Extract crop names and probabilities
                top_3_crops = [rec[0] for rec in top_3_recommendations]
                top_3_probs = [rec[1] for rec in top_3_recommendations]
                
                # Create DataFrame for top 3
                top_3_df = pd.DataFrame({
                    'Crop': top_3_crops,
                    'Confidence (%)': [round(prob, 2) for prob in top_3_probs]
                })
                
                # Display as table
                st.dataframe(
                    top_3_df.style.background_gradient(cmap='Greens', subset=['Confidence (%)']),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Bar chart for top 3
                fig = px.bar(
                    top_3_df,
                    x='Confidence (%)',
                    y='Crop',
                    orientation='h',
                    color='Confidence (%)',
                    color_continuous_scale='Greens',
                    title='Confidence Score Comparison'
                )
                fig.update_layout(
                    showlegend=False,
                    height=250,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Display input parameters
                st.subheader("📋 Your Input Parameters")
                
                param_col1, param_col2 = st.columns(2)
                
                with param_col1:
                    st.metric("Nitrogen (N)", f"{N} kg/ha")
                    st.metric("Phosphorus (P)", f"{P} kg/ha")
                    st.metric("Potassium (K)", f"{K} kg/ha")
                    st.metric("Temperature", f"{temperature}°C")
                
                with param_col2:
                    st.metric("Humidity", f"{humidity}%")
                    st.metric("Soil pH", f"{ph}")
                    st.metric("Rainfall", f"{rainfall} mm")
                
                # Get crop requirements if available
                if crop_data is not None:
                    crop_info = crop_data[crop_data['label'] == crop_name]
                    
                    if not crop_info.empty:
                        st.markdown("---")
                        st.subheader(f"📖 Ideal Requirements for {crop_name.title()}")
                        
                        req_col1, req_col2, req_col3 = st.columns(3)
                        
                        with req_col1:
                            st.info(f"**Nitrogen**\n\n{crop_info['N_avg'].values[0]:.1f} kg/ha (avg)")
                            st.info(f"**Phosphorus**\n\n{crop_info['P_avg'].values[0]:.1f} kg/ha (avg)")
                        
                        with req_col2:
                            st.info(f"**Potassium**\n\n{crop_info['K_avg'].values[0]:.1f} kg/ha (avg)")
                            st.info(f"**Temperature**\n\n{crop_info['temp_avg'].values[0]:.1f}°C (avg)")
                        
                        with req_col3:
                            st.info(f"**Humidity**\n\n{crop_info['humidity_avg'].values[0]:.1f}% (avg)")
                            st.info(f"**pH**\n\n{crop_info['ph_avg'].values[0]:.1f} (avg)")
                            
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
                st.error("Please check that all model files are properly loaded.")
        
        else:
            # Show placeholder
            st.info("👈 Enter your soil and climate parameters in the left panel and click 'Get Recommendation' to see results.")
            
            # Show some statistics
            st.subheader("📈 System Statistics")
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric("Total Crops", "22", help="Number of crops in database")
            
            with stat_col2:
                st.metric("Model Accuracy", "93.24%", help="SVM model accuracy")
            
            with stat_col3:
                st.metric("Training Samples", "8,800", help="Dataset size")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌾 <strong>Crop Recommendation System</strong> | Developed for SDG 2: Zero Hunger</p>
        <p>Helping farmers make data-driven decisions for sustainable agriculture</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# """
# Streamlit Web Application for Crop Recommendation System
# """

# import streamlit as st
# import pandas as pd
# import joblib
# import plotly.express as px
# import plotly.graph_objects as go
# from PIL import Image
# import os

# # Page configuration
# st.set_page_config(
#     page_title="Crop Recommendation System",
#     page_icon="🌾",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f0f8f0;
#     }
#     .stButton>button {
#         background-color: #2E7D32;
#         color: white;
#         font-size: 18px;
#         padding: 10px 24px;
#         border-radius: 8px;
#         border: none;
#     }
#     .stButton>button:hover {
#         background-color: #1B5E20;
#     }
#     .crop-card {
#         padding: 20px;
#         border-radius: 10px;
#         background-color: white;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin: 10px 0;
#     }
#     .metric-card {
#         background-color: #E8F5E9;
#         padding: 15px;
#         border-radius: 8px;
#         text-align: center;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Load CropPredictor
# @st.cache_resource
# def load_predictor():
#     """Load CropPredictor instance"""
#     try:
#         from src.models.predict import CropPredictor
#         predictor = CropPredictor()
#         return predictor
#     except Exception as e:
#         st.error(f"Error loading predictor: {e}")
#         st.error("Make sure src/models/predict.py exists with CropPredictor class")
#         return None
    
# # Load models
# @st.cache_resource
# def load_models():
#     """Load trained models and preprocessors"""
#     try:
#         model = joblib.load('models/crop_model_svm.pkl')
#         scaler = joblib.load('models/scaler.pkl')
#         encoder = joblib.load('models/label_encoder.pkl')
#         return model, scaler, encoder
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         return None, None, None

# # Load crop information
# @st.cache_data
# def load_crop_data():
#     """Load crop requirements data"""
#     try:
#         crop_req = pd.read_csv('data/processed/crop_requirements_summary.csv')
#         return crop_req
#     except:
#         return None

# # Prediction function
# def predict_crop(N, P, K, temperature, humidity, ph, rainfall, model, scaler, encoder):
#     """Make crop prediction"""
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
    
#     # Scale input
#     input_scaled = scaler.transform(input_data)
    
#     # Predict
#     prediction = model.predict(input_scaled)
#     probabilities = model.predict_proba(input_scaled)
    
#     # Get crop name
#     crop_name = encoder.inverse_transform(prediction)[0]
#     confidence = probabilities.max() * 100
    
#     # Get top 5 recommendations
#     top_5_idx = probabilities[0].argsort()[-5:][::-1]
#     top_5_crops = encoder.inverse_transform(top_5_idx)
#     top_5_probs = probabilities[0][top_5_idx] * 100
    
#     return crop_name, confidence, top_5_crops, top_5_probs

# #-------------------------------------------------
# # After prediction
# explanations = predictor.explain_prediction(input_params, crop_name, crop_data)

# st.subheader("🔍 Why This Crop?")
# for explanation in explanations:
#     st.write(explanation)

# #------------------------------------------------------

# # Main app
# def main():
#     # Load models
#     model, scaler, encoder = load_models()
#     crop_data = load_crop_data()
    
#     if model is None:
#         st.error("Failed to load models. Please check if model files exist in 'models/' directory.")
#         return
    
#     # Header
#     st.title("🌾 Crop Recommendation System")
#     st.markdown("### Make data-driven decisions for optimal crop selection")
#     st.markdown("---")
    
#     # Sidebar
#     st.sidebar.header("📊 About")
#     st.sidebar.info("""
#     This intelligent system recommends the best crop to plant based on:
#     - Soil nutrients (NPK)
#     - Climate conditions
#     - Soil pH
#     - Rainfall patterns
    
#     **Accuracy: 93.24%**
#     **Model: Support Vector Machine (SVM)**
#     """)
    
#     st.sidebar.markdown("---")
#     st.sidebar.header("🎯 SDG Impact")
#     st.sidebar.success("""
#     **SDG 2: Zero Hunger**
#     - Increase yields by 15-25%
#     - Reduce fertilizer waste by 30%
#     - Support sustainable agriculture
#     """)
    
#     # Main content - Two columns
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("📝 Input Soil & Climate Parameters")
        
#         with st.form("prediction_form"):
#             # Nutrient inputs
#             st.subheader("🧪 Soil Nutrients")
#             col_n, col_p, col_k = st.columns(3)
            
#             with col_n:
#                 N = st.number_input(
#                     "Nitrogen (N) kg/ha",
#                     min_value=0,
#                     max_value=140,
#                     value=90,
#                     help="Nitrogen content in soil (0-140 kg/ha)"
#                 )
            
#             with col_p:
#                 P = st.number_input(
#                     "Phosphorus (P) kg/ha",
#                     min_value=5,
#                     max_value=145,
#                     value=42,
#                     help="Phosphorus content in soil (5-145 kg/ha)"
#                 )
            
#             with col_k:
#                 K = st.number_input(
#                     "Potassium (K) kg/ha",
#                     min_value=5,
#                     max_value=205,
#                     value=43,
#                     help="Potassium content in soil (5-205 kg/ha)"
#                 )
            
#             st.markdown("---")
            
#             # Climate inputs
#             st.subheader("🌡️ Climate Conditions")
#             col_temp, col_hum = st.columns(2)
            
#             with col_temp:
#                 temperature = st.slider(
#                     "Temperature (°C)",
#                     min_value=8.0,
#                     max_value=44.0,
#                     value=21.0,
#                     step=0.5,
#                     help="Average temperature (8-44°C)"
#                 )
            
#             with col_hum:
#                 humidity = st.slider(
#                     "Humidity (%)",
#                     min_value=14.0,
#                     max_value=100.0,
#                     value=82.0,
#                     step=1.0,
#                     help="Relative humidity (14-100%)"
#                 )
            
#             st.markdown("---")
            
#             # Soil pH and Rainfall
#             st.subheader("🌧️ Additional Parameters")
#             col_ph, col_rain = st.columns(2)
            
#             with col_ph:
#                 ph = st.slider(
#                     "Soil pH",
#                     min_value=3.5,
#                     max_value=9.9,
#                     value=6.5,
#                     step=0.1,
#                     help="Soil pH level (3.5-9.9)"
#                 )
            
#             with col_rain:
#                 rainfall = st.slider(
#                     "Rainfall (mm)",
#                     min_value=20.0,
#                     max_value=300.0,
#                     value=202.0,
#                     step=5.0,
#                     help="Annual rainfall (20-300mm)"
#                 )
            
#             st.markdown("---")
            
#             # Submit button
#             submitted = st.form_submit_button("🔍 Get Recommendation", use_container_width=True)
    
#     with col2:
#         st.header("🎯 Recommendation Results")
        
#         if submitted:
#         # Make prediction
#             result = predict_crop(
#             N, P, K, temperature, humidity, ph, rainfall,
#             model, scaler, encoder
#     )
    
#             crop_name = result['recommended_crop']
#             confidence = result['confidence']
#             top_5_crops = result['top_5_crops']
#             top_5_probs = result['top_5_probs']
    
#     # Get explanations and warnings
#             from src.models.predict import CropPredictor
#             predictor = CropPredictor()
    
#             input_params = {
#                 'N': N, 'P': P, 'K': K,
#                 'temperature': temperature,
#                 'humidity': humidity,
#                 'ph': ph,
#                 'rainfall': rainfall
#             }
            
#             explanations = predictor.explain_prediction(input_params, crop_name)
#             warnings = predictor.check_edge_cases(input_params)
            
#             # Display warnings first (if any)
#             if warnings:
#                 st.warning("### ⚠️ Important Alerts")
#                 for warning in warnings:
#                     st.markdown(warning)
#                 st.markdown("---")
            
#             # Display main recommendation
#             st.markdown(f"""
#             <div class="crop-card">
#                 <h1 style='text-align: center; color: #2E7D32;'>🌾 {crop_name.upper()}</h1>
#                 <h3 style='text-align: center; color: #666;'>Recommended Crop</h3>
#                 <div class="metric-card">
#                     <h2 style='color: #1B5E20; margin: 0;'>{confidence:.2f}%</h2>
#                     <p style='margin: 5px 0 0 0;'>Confidence Score</p>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Progress bar for confidence
#             st.progress(confidence / 100)
            
#             st.markdown("---")
            
#             # NEW: Explanation Section
#             st.subheader("🔍 Why This Crop?")
#             st.info("Here's how your soil and climate conditions match this crop's requirements:")
            
#             for explanation in explanations:
#                 if "✅" in explanation:
#                     st.success(explanation)
#                 elif "✓" in explanation:
#                     st.info(explanation)
#                 else:
#                     st.warning(explanation)
            
#             st.markdown("---")
            
#             # Top 5 recommendations (existing code continues...)
#             st.subheader("📊 Top 5 Alternative Crops")
    
#     #--------------------------------------------
#             # Create DataFrame for top 5
#             top_5_df = pd.DataFrame({
#                 'Crop': top_5_crops,
#                 'Confidence (%)': top_5_probs.round(2)
#             })
            
#             # Display as table
#             st.dataframe(
#                 top_5_df.style.background_gradient(cmap='Greens', subset=['Confidence (%)']),
#                 use_container_width=True,
#                 hide_index=True
#             )
            
#             # Bar chart for top 5
#             fig = px.bar(
#                 top_5_df,
#                 x='Confidence (%)',
#                 y='Crop',
#                 orientation='h',
#                 color='Confidence (%)',
#                 color_continuous_scale='Greens',
#                 title='Confidence Score Comparison'
#             )
#             fig.update_layout(
#                 showlegend=False,
#                 height=300,
#                 margin=dict(l=0, r=0, t=40, b=0)
#             )
#             st.plotly_chart(fig, use_container_width=True)
            
#             st.markdown("---")
            
#             # Display input parameters
#             st.subheader("📋 Your Input Parameters")
            
#             param_col1, param_col2 = st.columns(2)
            
#             with param_col1:
#                 st.metric("Nitrogen (N)", f"{N} kg/ha")
#                 st.metric("Phosphorus (P)", f"{P} kg/ha")
#                 st.metric("Potassium (K)", f"{K} kg/ha")
#                 st.metric("Temperature", f"{temperature}°C")
            
#             with param_col2:
#                 st.metric("Humidity", f"{humidity}%")
#                 st.metric("Soil pH", f"{ph}")
#                 st.metric("Rainfall", f"{rainfall} mm")
            
#             # Get crop requirements if available
#             if crop_data is not None:
#                 crop_info = crop_data[crop_data['crop'] == crop_name]
                
#                 if not crop_info.empty:
#                     st.markdown("---")
#                     st.subheader(f"📖 Ideal Requirements for {crop_name.title()}")
                    
#                     req_col1, req_col2, req_col3 = st.columns(3)
                    
#                     with req_col1:
#                         st.info(f"**Nitrogen**\n\n{crop_info['N_avg'].values[0]:.1f} kg/ha (avg)")
#                         st.info(f"**Phosphorus**\n\n{crop_info['P_avg'].values[0]:.1f} kg/ha (avg)")
                    
#                     with req_col2:
#                         st.info(f"**Potassium**\n\n{crop_info['K_avg'].values[0]:.1f} kg/ha (avg)")
#                         st.info(f"**Temperature**\n\n{crop_info['temp_avg'].values[0]:.1f}°C (avg)")
                    
#                     with req_col3:
#                         st.info(f"**Humidity**\n\n{crop_info['humidity_avg'].values[0]:.1f}% (avg)")
#                         st.info(f"**pH**\n\n{crop_info['ph_avg'].values[0]:.1f} (avg)")
        
#         else:
#             # Show placeholder
#             st.info("👈 Enter your soil and climate parameters in the left panel and click 'Get Recommendation' to see results.")
            
#             # Show some statistics
#             st.subheader("📈 System Statistics")
            
#             stat_col1, stat_col2, stat_col3 = st.columns(3)
            
#             with stat_col1:
#                 st.metric("Total Crops", "22", help="Number of crops in database")
            
#             with stat_col2:
#                 st.metric("Model Accuracy", "93.24%", help="SVM model accuracy")
            
#             with stat_col3:
#                 st.metric("Training Samples", "8,800", help="Dataset size")
    
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style='text-align: center; color: #666;'>
#         <p>🌾 <strong>Crop Recommendation System</strong> | Developed for SDG 2: Zero Hunger</p>
#         <p>Helping farmers make data-driven decisions for sustainable agriculture</p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()