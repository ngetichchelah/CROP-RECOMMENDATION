"""
Streamlit Web Application for Crop Recommendation System
WITH ENHANCEMENTS: Tabbed Interface + Explainability + Edge Case Warnings + Resource Filters + OOD Detection
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#=======================for CF:=================================
from scripts.hybrid_recommender import HybridRecommender
from utils.helpers import append_interaction, append_rating, initialize_csv_files, get_ratings_count

# Initialize CSV files on startup
initialize_csv_files()

# Page configuration
st.set_page_config(
    page_title="Intelligent Crop Recommendation System (I.C.R.S)",
    page_icon="🌾",
    layout="wide",
    #initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f8f0;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: black;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    
    .crop-card {
        padding: 10px;
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
    .ood-warning {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #2E7D32;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        color: white !important; /* Make text white */
        cursor: default; /* Disable pointer cursor on hover */
        transition: none; /* Disable hover animation */
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2E7D32;
        color: white;
    }
    
    /* Disable hover effect completely */
    .stTabs [data-baseweb="tab"]:hover {
    background-color: #2E7D32 !important;
    color: white !important;
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

# Load crop information (requirements)
@st.cache_data
def load_crop_data():
    """Load crop requirements data"""
    try:
        crop_req = pd.read_csv('data/processed/crop_requirements_summary.csv')
        return crop_req
    except Exception as e:
        st.warning(f"Could not load crop requirements: {e}")
        return None

# Load resource data (cost, labour, irrigation)
@st.cache_data
def load_resource_data():
    """Load crop resource requirements"""
    try:
        return pd.read_csv('data/crop_resources.csv')
    except Exception as e:
        st.warning(f"Could not load resource data: {e}")
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

# Load training data for OOD detection
@st.cache_data
def load_training_stats():
    """Load training data statistics for OOD detection"""
    try:
        df_train = pd.read_csv('data/processed/crop_data_cleaned.csv')
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        stats = {}
        for feature in features:
            stats[feature] = {
                'mean': df_train[feature].mean(),
                'std': df_train[feature].std(),
                'min': df_train[feature].min(),
                'max': df_train[feature].max()
            }
        return stats
    except Exception as e:
        st.warning(f"Could not load training statistics: {e}")
        return None

# Out-of-Distribution Input Detection
def check_ood_inputs(params, stats):
    """
    Check if inputs are out-of-distribution (unusual)
    Returns list of warnings for unusual parameters
    """
    if stats is None:
        return []
    
    warnings = []
    
    for param, value in params.items():
        if param in stats:
            mean = stats[param]['mean']
            std = stats[param]['std']
            z_score = abs(value - mean) / std if std > 0 else 0
            
            # Flag if z-score > 2.5 (unusual value)
            if z_score > 2.5:
                warnings.append({
                    'param': param,
                    'value': value,
                    'z_score': z_score,
                    'expected_range': f"{mean - 2*std:.1f} - {mean + 2*std:.1f}",
                    'message': f"{param} = {value} is unusual (z-score: {z_score:.1f})"
                })
    
    return warnings

# Filter recommended crops based on user resources
def filter_by_resources(predictions, constraints, resource_data):
    """
    Filter crop recommendations by farmer's resource constraints
    
    predictions: list of (crop, confidence) tuples
    constraints: dict with budget, labor, irrigation, max_wait
    resource_data: DataFrame with crop resource requirements
    
    Returns: (feasible_crops, excluded_crops)
    """
    feasible = []
    excluded = []
    
    for crop, confidence in predictions:
        # Get crop resource requirements
        crop_info = resource_data[resource_data['crop'] == crop]
        
        if crop_info.empty:
            # Unknown crop, allow by default
            feasible.append((crop, confidence))
            continue
        
        crop_info = crop_info.iloc[0]
        total_cost = crop_info['seed_cost_usd'] + crop_info['fertilizer_cost_usd']
        labor_needed = crop_info['labor_days']
        needs_irrigation = crop_info['irrigation_needed']
        harvest_time = crop_info['harvest_months']
        
        # Check constraints and store reasons if excluded
        reasons = []
        
        if total_cost > constraints['max_budget']:
            reasons.append(f"💰 Cost ${total_cost:.0f} exceeds budget ${constraints['max_budget']}")
        
        if labor_needed > constraints['max_labor']:
            reasons.append(f"👷 Needs {labor_needed} labor days (you have {constraints['max_labor']})")
        
        if needs_irrigation and not constraints['irrigation']:
            reasons.append(f"💧 Requires irrigation (unavailable)")
        
        if harvest_time > constraints['max_wait']:
            reasons.append(f"⏱️ Harvest in {harvest_time} months (wait limit: {constraints['max_wait']} months)")
        
        if reasons:
            excluded.append((crop, confidence, reasons))
        else:
            feasible.append((crop, confidence))
    
    return feasible, excluded

# Main app
def main():
    # Load predictor and data
    predictor = load_predictor()
    crop_data = load_crop_data()
    resource_data = load_resource_data()
    training_stats = load_training_stats()
    
    # If predictor fails, show error and exit
    if predictor is None:
        st.error("❌ Failed to load predictor. Please check:")
        st.error("1. models/ folder exists with crop_model_svm.pkl, scaler.pkl, label_encoder.pkl")
        st.error("2. src/models/predict.py exists with CropPredictor class")
        return
    
    # Header
    st.title("🌾 Intelligent Crop Recommendation System")
    st.markdown("📊 Make data-driven decisions for optimal crop selection")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📊 About")
    st.sidebar.info("""
    This intelligent system recommends the best crop to plant based on:
    - Soil nutrients (NPK)
    - Climate conditions
    - Soil pH
    - Rainfall patterns
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 SDG Impact")
    st.sidebar.success("""
    **SDG 2: Zero Hunger**
    - Increase yields
    - Reduce fertilizer waste
    - Support sustainable agriculture
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚠️ Important Notes")
    st.sidebar.warning("""
    **Regional Validation:**
    This system has been validated on global agricultural data. 
    For deployment in specific regions, local calibration is recommended.
    
    **Resource Requirements:**
    Use the resource filters to ensure recommendations 
    match your available resources.
    """)
    
    # Add help button
    if st.sidebar.button("🤝 Get Help with Resources"):
        st.sidebar.info("""
        ### Where to Get Support:
        
        **Kenya:**
        - Ministry of Agriculture Extension Services
        - Kenya Agricultural Research Organization (KALRO)
        - One Acre Fund: Free inputs + training
        
        **Financial Support:**
        - Kilimo Biashara Fund: Low-interest loans
        - County agricultural funds
        - Group lending through cooperatives
        """)
    
    # Initialize session state for storing prediction results
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'input_params' not in st.session_state:
        st.session_state.input_params = None
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Input & Predict",
        "📊 Recommendations",
        "🌦️ Climate Analysis",
        "💰 Resource Planning",
        "📈 System Insights"
    ])
    
    # ============================================
    # TAB 1: INPUT & PREDICT
    # ============================================
    with tab1:
        st.header("📝 Enter Your Farm Parameters")
        
        with st.form("prediction_form"):
            # Nutrient inputs
            st.subheader("🧪 Soil Nutrients")
            col_n, col_p, col_k = st.columns(3)
            
            with col_n:
                N = st.number_input(
                    "Nitrogen (N) kg/ha",
                    min_value=0,
                    max_value=140,
                    value=0,
                    help="Nitrogen content in soil (0-140 kg/ha)"
                )
            
            with col_p:
                P = st.number_input(
                    "Phosphorus (P) kg/ha",
                    min_value=5,
                    max_value=145,
                    value=5,
                    help="Phosphorus content in soil (5-145 kg/ha)"
                )
            
            with col_k:
                K = st.number_input(
                    "Potassium (K) kg/ha",
                    min_value=5,
                    max_value=205,
                    value=5,
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
                    value=8.0,
                    step=0.5,
                    help="Average temperature (8-44°C)"
                )
            
            with col_hum:
                humidity = st.slider(
                    "Humidity (%)",
                    min_value=14.0,
                    max_value=100.0,
                    value=14.0,
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
                    value=3.5,
                    step=0.1,
                    help="Soil pH level (3.5-9.9)"
                )
            
            with col_rain:
                rainfall = st.slider(
                    "Rainfall (mm)",
                    min_value=20.0,
                    max_value=300.0,
                    value=20.0,
                    step=5.0,
                    help="Annual rainfall (20-300mm)"
                )
            
            st.markdown("---")
            
            # === RESOURCE CONSTRAINT SECTION ===
            st.subheader("💰 Resource Constraints (Optional)")
            st.caption("Help us recommend crops you can actually afford and manage")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                max_budget = st.number_input(
                    "Maximum budget (USD)",
                    min_value=0,
                    max_value=500,
                    value=0,
                    step=10,
                    help="Total for seeds + fertilizer"
                )

                max_labor = st.slider(
                    "Labor days available",
                    min_value=10,
                    max_value=150,
                    value=0,
                    help="Person-days per season"
                )

            with res_col2:
                irrigation = st.radio(
                    "Irrigation available?",
                    options=["Yes", "No"],
                    index=1,
                    help="Can you provide supplemental water?"
                )
                irrigation_bool = (irrigation == "Yes")

                max_wait = st.selectbox(
                    "Maximum wait to harvest",
                    options=[3, 4, 6, 12, 24, 36],
                    index=2,
                    format_func=lambda x: f"{x} months",
                    help="Cash flow constraint"
                )

            # Checkbox toggle for applying filters
            #use_constraints = st.checkbox("Apply resource constraints", value=False)
            
            # Checkbox toggle for applying filters
            use_constraints = st.checkbox("Apply resource constraints", value=False)

            # Only collect constraints if user wants to apply them
            if use_constraints:
                constraints = {
                    'max_budget': max_budget,
                    'max_labor': max_labor,
                    'irrigation': irrigation_bool,
                    'max_wait': max_wait
                }
            else:
                constraints = None

            # Later in the submission section, replace this:
            st.session_state.constraints = {
                'max_budget': max_budget,
                'max_labor': max_labor,
                'irrigation': irrigation_bool,
                'max_wait': max_wait
            } if use_constraints else None

            # With this:
            st.session_state.constraints = constraints
            st.markdown("---")
            
            # Optional: User location for regional tracking
            user_location = st.text_input(
                "Your location (optional)",
                placeholder="e.g., Makueni County, Kenya",
                help="Helps us understand regional performance"
            )
            
            st.markdown("---")
            
            # Submit button - MUST BE INSIDE FORM
            submitted = st.form_submit_button("🔍 Get Recommendation", use_container_width=True)
        
        # Process submission
        if submitted:
            # Prepare input parameters
            input_params = {
                'N': N,
                'P': P,
                'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
            
            # === OOD DETECTION ===
            ood_warnings = check_ood_inputs(input_params, training_stats)
            
            if ood_warnings:
                # Display warning for unusual inputs
                st.markdown("""
                <div class="ood-warning">
                    <h3>⚠️ UNUSUAL INPUT DETECTED</h3>
                    <p>Your soil/climate conditions are significantly different from typical values. 
                    This may indicate:</p>
                    <ul>
                        <li>Model not validated for your region</li>
                        <li>Measurement error in soil test</li>
                        <li>Extreme environmental conditions</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.warning("**Unusual parameters:**")
                for w in ood_warnings:
                    st.write(f"• **{w['param']}** = {w['value']} (z-score: {w['z_score']:.1f})")
                    st.write(f"  Expected range: {w['expected_range']}")
                
                st.info("""
                **Recommendations:**
                - ✅ Verify your measurements (check soil test accuracy)
                - ✅ Consult local agricultural officer before planting
                - ✅ Consider these predictions preliminary
                - ⚠️ System may need regional calibration for your area
                """)
            
            # Make prediction using CropPredictor
            try:
                result = predictor.recommend_crop(
                    N=N, P=P, K=K,
                    temperature=temperature,
                    humidity=humidity,
                    ph=ph,
                    rainfall=rainfall
                )
                
                # Store in session state
                st.session_state.prediction_made = True
                st.session_state.result = result
                st.session_state.input_params = input_params
                st.session_state.ood_warnings = ood_warnings
                st.session_state.use_constraints = use_constraints
                st.session_state.constraints = {
                    'max_budget': max_budget,
                    'max_labor': max_labor,
                    'irrigation': irrigation_bool,
                    'max_wait': max_wait
                } if use_constraints else None
                st.session_state.user_location = user_location
                
                st.success("✅ Prediction completed! View results in the other tabs.")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
                st.error("Please check that all model files are properly loaded.")
        
        # Show placeholder when no prediction
        if not st.session_state.prediction_made:
            st.info("👆 Fill in your farm parameters above and click 'Get Recommendation' to see results in the other tabs.")
    
    # ============================================
    # TAB 2: RECOMMENDATIONS
    # ============================================
    with tab2:
        if not st.session_state.prediction_made:
            st.info("👈 Please make a prediction in the 'Input & Predict' tab first.")
        else:
            result = st.session_state.result
            crop_name = result['recommended_crop']
            confidence = result['confidence']
            top_3_recommendations = result['top_3_recommendations']
            explanations = result['explanations']
            warnings = result['warnings']
            ood_warnings = st.session_state.get('ood_warnings', [])
            
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
            
            # Adjust confidence if OOD
            if ood_warnings:
                adjusted_conf = confidence * 0.7
                st.warning(f"""
                **⚠️ Confidence Adjusted for Unusual Inputs:**
                - Original: {confidence:.1f}%
                - Adjusted: {adjusted_conf:.1f}%
                """)
                confidence = adjusted_conf
            
            # Progress bar for confidence
            st.progress(confidence / 100)
            
            st.markdown("---")
            
            # === EXPLANATION SECTION ===
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
            
            # === RESOURCE-FILTERED RECOMMENDATIONS ===
            if st.session_state.use_constraints and st.session_state.constraints is not None:
            #if st.session_state.use_constraints:
                constraints = st.session_state.constraints
                top_5 = [(crop, conf) for crop, conf in result.get('all_predictions', top_3_recommendations)[:5]]
                feasible, excluded = filter_by_resources(top_5, constraints, resource_data)
                
                if len(excluded) > 0:
                    st.info(f"""
                    📊 **Resource Filter Applied:**
                    - {len(feasible)} crops match your resources
                    - {len(excluded)} crops excluded due to constraints
                    """)
                
                if feasible:
                    st.subheader("✅ Crops Matching Your Resources")
                    
                    for i, (crop, conf) in enumerate(feasible[:3], 1):
                        crop_res = resource_data[resource_data['crop'] == crop]
                        
                        if not crop_res.empty:
                            crop_res = crop_res.iloc[0]
                            total_cost = crop_res['seed_cost_usd'] + crop_res['fertilizer_cost_usd']
                            
                            with st.expander(f"{i}. {crop.upper()} - {conf:.1f}% confidence"):
                                res_col1, res_col2, res_col3 = st.columns(3)
                                
                                with res_col1:
                                    st.metric("Total Cost", f"${total_cost:.0f}")
                                    st.metric("Labor", f"{crop_res['labor_days']} days")
                                
                                with res_col2:
                                    st.metric("Harvest Time", f"{crop_res['harvest_months']} months")
                                    st.metric("Irrigation", "Yes" if crop_res['irrigation_needed'] else "No")
                                
                                with res_col3:
                                    st.metric("Market Access", crop_res['market_access'])
                                
                                st.success(f"""
                                **Why this works for you:**
                                - ✅ Fits ${constraints['max_budget']} budget (costs ${total_cost:.0f})
                                - ✅ Manageable labor ({crop_res['labor_days']} ≤ {constraints['max_labor']} days)
                                - ✅ {'Has' if constraints['irrigation'] else 'No'} irrigation ({'needed' if crop_res['irrigation_needed'] else 'not needed'})
                                - ✅ Harvest in {crop_res['harvest_months']} months (≤ {constraints['max_wait']} month limit)
                                """)
                        else:
                            st.write(f"{i}. **{crop.upper()}** - {conf:.1f}% confidence")
                
                elif not feasible:
                    st.error("❌ No crops match all your constraints")
                    st.info("""
                    **Suggestions:**
                    - 💰 Increase budget (seek credit/subsidies)
                    - 👷 Extend labor availability (hire help)
                    - 💧 Consider irrigation investment
                    - ⏱️ Extend harvest wait time
                    """)
                
                # Show excluded crops
                if excluded:
                    with st.expander(f"📋 {len(excluded)} crops excluded (click to see why)"):
                        for crop, conf, reasons in excluded:
                            st.warning(f"**{crop.upper()}** ({conf:.1f}% confidence)")
                            for reason in reasons:
                                st.write(f"  {reason}")
            
            #===========added=====================
            else:
                st.info("💡 **Tip:** Enable resource constraints in the Input tab to get personalized recommendations based on your budget, labor, and irrigation availability.")
            
            st.markdown("---")
            
            # === TOP 3 ALTERNATIVES ===
            st.subheader("📊 Top 3 Alternative Crops (By Suitability)")
            
            top_3_crops = [rec[0] for rec in top_3_recommendations]
            top_3_probs = [rec[1] for rec in top_3_recommendations]
            
            top_3_df = pd.DataFrame({
                'Crop': top_3_crops,
                'Confidence (%)': [round(prob, 2) for prob in top_3_probs]
            })
            
            st.dataframe(
                top_3_df.style.background_gradient(cmap='Greens', subset=['Confidence (%)']),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            
            # === INPUT PARAMETERS ===
            st.subheader("📋 Your Input Parameters")
            
            input_params = st.session_state.input_params
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                st.metric("Nitrogen (N)", f"{input_params['N']} kg/ha")
                st.metric("Phosphorus (P)", f"{input_params['P']} kg/ha")
                st.metric("Potassium (K)", f"{input_params['K']} kg/ha")
                st.metric("Temperature", f"{input_params['temperature']}°C")
            
            with param_col2:
                st.metric("Humidity", f"{input_params['humidity']}%")
                st.metric("Soil pH", f"{input_params['ph']}")
                st.metric("Rainfall", f"{input_params['rainfall']} mm")
            
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
    
    # ============================================
    # TAB 3: CLIMATE ANALYSIS
    # ============================================
    with tab3:
        if not st.session_state.prediction_made:
            st.info("👈 Please make a prediction in the 'Input & Predict' tab first.")
        else:
            result = st.session_state.result
            scenarios = result.get('scenarios', {})
            
            st.header("🌦️ Climate Scenario Analysis")
            
            if scenarios:
                # Primary scenario
                if 'type' in scenarios and scenarios['type'] in ['drought', 'flood']:
                    scenario = scenarios
                    
                    if scenario['alert_level'] == 'warning':
                        st.warning(f"### {scenario['title']}")
                    elif scenario['alert_level'] == 'error':
                        st.error(f"### {scenario['title']}")
                    else:
                        st.info(f"### {scenario['title']}")
                    
                    st.write(f"**{scenario['description']}**")
                    st.write(f"**Recommendation**: {scenario['recommendation']}")
                    
                    if 'alternative_crops' in scenario:
                        st.info(f"**🌾 Alternative crops**: {', '.join(scenario['alternative_crops'])}")
                    
                    if 'advice' in scenario:
                        with st.expander("📋 Detailed Farming Recommendations", expanded=True):
                            for advice_item in scenario['advice']:
                                st.write(f"• {advice_item}")
                    
                    st.markdown("---")
                
                # Secondary scenarios
                secondary_scenarios = [
                    scenarios.get('heat'),
                    scenarios.get('cold'),
                    scenarios.get('arid'),
                    scenarios.get('tropical')
                ]
                
                for scenario in secondary_scenarios:
                    if scenario:
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
                        
                        if 'advice' in scenario:
                            with st.expander(f"View recommendations for {scenario['type']} conditions"):
                                for advice_item in scenario['advice']:
                                    st.write(f"• {advice_item}")
            else:
                st.info("No specific climate scenarios detected for your input parameters.")
    
    # ============================================
    # TAB 4: RESOURCE PLANNING
    # ============================================
    with tab4:
        if not st.session_state.prediction_made:
            st.info("👈 Please make a prediction in the 'Input & Predict' tab first.")
        else:
            result = st.session_state.result
            crop_name = result['recommended_crop']
            
            st.header("💰 Resource Planning & Economic Analysis")
            
            # Get crop resource info
            crop_res = resource_data[resource_data['crop'] == crop_name]
            
            if not crop_res.empty:
                crop_res = crop_res.iloc[0]
                total_cost = crop_res['seed_cost_usd'] + crop_res['fertilizer_cost_usd']
                
                # Cost breakdown
                st.subheader("💵 Cost Breakdown")
                
                cost_col1, cost_col2, cost_col3 = st.columns(3)
                
                with cost_col1:
                    st.metric("Seed Cost", f"${crop_res['seed_cost_usd']:.0f}")
                
                with cost_col2:
                    st.metric("Fertilizer Cost", f"${crop_res['fertilizer_cost_usd']:.0f}")
                
                with cost_col3:
                    st.metric("Total Cost", f"${total_cost:.0f}", 
                             delta=None,
                             delta_color="normal")
                
                st.markdown("---")
                
                # Labor & Time requirements
                st.subheader("👷 Labor & Time Requirements")
                
                labor_col1, labor_col2, labor_col3 = st.columns(3)
                
                with labor_col1:
                    st.metric("Labor Days", f"{crop_res['labor_days']} days")
                
                with labor_col2:
                    st.metric("Harvest Time", f"{crop_res['harvest_months']} months")
                
                with labor_col3:
                    st.metric("Irrigation Needed", "Yes" if crop_res['irrigation_needed'] else "No")
                
                st.markdown("---")
                
                # Market access
                st.subheader("📍 Market Access")
                
                market_colors = {
                    'HIGH': '🟢',
                    'MEDIUM': '🟡',
                    'LOW': '🔴'
                }
                
                st.info(f"""
                **Market Access Level**: {market_colors.get(crop_res['market_access'], '⚪')} {crop_res['market_access']}
                
                {'✅ Good market accessibility - easy to sell produce' if crop_res['market_access'] == 'HIGH' else '⚠️ Limited market access - may need transport arrangements' if crop_res['market_access'] == 'MEDIUM' else '❌ Low market access - requires significant logistics planning'}
                """)
                
                st.markdown("---")
                
                # Financial planning tips
                st.subheader("💡 Financial Planning Tips")
                
                if total_cost > 150:
                    st.warning(f"""
                    **⚠️ High-Investment Crop Detected**
                    
                    {crop_name.title()} requires significant upfront investment (${total_cost:.0f}).
                    
                    **Financing Options:**
                    - 🏦 Agricultural loans (check with local banks)
                    - 👥 Group lending through cooperatives
                    - 🎁 Government subsidies for inputs
                    - 📊 Phased planting (start small, expand gradually)
                    """)
                else:
                    st.success(f"""
                    **✅ Affordable Crop Option**
                    
                    {crop_name.title()} has moderate input costs (${total_cost:.0f}).
                    This makes it accessible for small-scale farmers.
                    """)
                
                if crop_res['harvest_months'] > 6:
                    st.info(f"""
                    **⏰ Long-Term Crop (Harvest: {crop_res['harvest_months']} months)**
                    
                    Consider intercropping with short-season crops for cash flow during the wait period.
                    """)
                
                st.markdown("---")
                
                # Resource comparison with other top crops
                st.subheader("📊 Compare with Alternative Crops")
                
                top_3_crops = [rec[0] for rec in result['top_3_recommendations']]
                comparison_data = []
                
                for crop in top_3_crops:
                    crop_info = resource_data[resource_data['crop'] == crop]
                    if not crop_info.empty:
                        crop_info = crop_info.iloc[0]
                        comparison_data.append({
                            'Crop': crop.title(),
                            'Total Cost ($)': crop_info['seed_cost_usd'] + crop_info['fertilizer_cost_usd'],
                            'Labor (days)': crop_info['labor_days'],
                            'Harvest (months)': crop_info['harvest_months'],
                            'Irrigation': 'Yes' if crop_info['irrigation_needed'] else 'No'
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(
                        comparison_df.style.background_gradient(cmap='RdYlGn_r', subset=['Total Cost ($)', 'Labor (days)', 'Harvest (months)']),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.warning(f"Resource data not available for {crop_name}")
            
            st.markdown("---")
            
            # === REGIONAL VALIDATION STATUS ===
            user_location = st.session_state.get('user_location', '')
            
            if user_location:
                st.subheader("🗺️ Regional Validation Status")
                
                # Check if region is validated (placeholder - would connect to database)
                validated_regions = ["Global Dataset", "India", "Bangladesh"]
                
                if any(region.lower() in user_location.lower() for region in validated_regions):
                    st.success(f"""
                    ✅ **System Validated for Similar Regions**
                    
                    Your location ({user_location}) falls within regions covered by our training data.
                    Recommendations should be reasonably accurate.
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **System Not Yet Validated in {user_location}**
                    
                    We have limited data from your region. Recommendations should be
                    considered preliminary until local validation is completed.
                    
                    **How to help:**
                    - Plant recommended crop and report your yield
                    - Join pilot study (contact: pilot@croprec.org)
                    - Share with local agricultural officers
                    
                    **Regions with validation data:**
                    {', '.join(validated_regions)}
                    """)
                    
                    st.info("""
                    **Validation Protocol:**
                    Before full deployment, we require:
                    1. ✅ Desktop validation (climate range check)
                    2. ⏳ Retrospective validation (historical data, >85% accuracy)
                    3. ⏳ Prospective pilot (100 farmers, 6 months)
                    """)
            
            st.markdown("---")
            
            # === EXPERT FEEDBACK SYSTEM ===
            st.subheader(" Expert Feedback (Optional)")
            
            expert_mode = st.checkbox("I am an agricultural expert", value=False)
            
            if expert_mode:
                expert_opinion = st.radio(
                    "Do you agree with this recommendation?",
                    options=["Agree", "Disagree", "Uncertain"],
                    help="Your feedback helps improve the system"
                )
                
                if expert_opinion == "Disagree":
                    expert_reason = st.text_area(
                        "Why do you disagree?",
                        placeholder="e.g., This crop doesn't grow well in this region due to..."
                    )
                    
                    expert_alternative = st.selectbox(
                        "What would you recommend instead?",
                        options=sorted(['rice', 'maize', 'chickpea', 'cotton', 'coffee', 
                                      'jute', 'kidneybeans', 'pigeonpeas', 'mothbeans',
                                      'mungbean', 'blackgram', 'lentil', 'pomegranate',
                                      'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
                                      'apple', 'papaya', 'coconut', 'orange', 'groundnuts'])
                    )
                    
                    if st.button("Submit Expert Feedback"):
                        # In production, this would save to database
                        
                        # Create feedback record
                        feedback_data = {
                            'timestamp': pd.Timestamp.now(),
                            'user_location': user_location,
                            'model_recommendation': crop_name,
                            'model_confidence': result['confidence'],
                            'expert_agreement': expert_opinion,
                            'expert_alternative': expert_alternative if expert_opinion == "Disagree" else None,
                            'expert_reason': expert_reason if expert_opinion == "Disagree" else None,
                            'input_parameters': st.session_state.input_params
                        }
                        
                        # Save to CSV file (can be database later)
                        try:
                            import os
                            os.makedirs('data/feedback', exist_ok=True)
                            feedback_file = 'data/feedback/expert_feedback.csv'
                            
                            # Create DataFrame and save
                            feedback_df = pd.DataFrame([feedback_data])
                            
                            if os.path.exists(feedback_file):
                                # Append to existing file
                                existing_df = pd.read_csv(feedback_file)
                                updated_df = pd.concat([existing_df, feedback_df], ignore_index=True)
                                updated_df.to_csv(feedback_file, index=False)
                            else:
                                # Create new file
                                feedback_df.to_csv(feedback_file, index=False)
                             
                             #user_location = feedback_file   
                                            
                        
                            st.success("""
                            ✅ **Thank you!** Your feedback will be used to:
                            1. Flag this recommendation for review
                            2. Improve future model versions
                            3. Build regional calibration data
                            
                            Feedback ID: EXP-{}-001 (for your records)
                            """.format(user_location[:3].upper() if user_location else "GLB"))
                            
                        except Exception as e:
                            st.error(f"❌ Failed to save feedback: {e}")
                            
                        # Log feedback (placeholder)
                        st.info(f"""
                        **Logged:**
                        - Location: {feedback_file or 'Not specified'}
                        - Model recommendation: {crop_name} ({result['confidence']:.1f}%)
                        - Expert recommendation: {expert_alternative}
                        - Reason: {expert_reason[:100]}...
                        """)
    
    # ============================================
    # TAB 5: SYSTEM INSIGHTS
    # ============================================
    with tab5:
        st.header("📈 System Statistics & Performance")
        
        # Show system statistics
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Total Crops", "22", help="Number of crops in database")
        
        with stat_col2:
            st.metric("Model Accuracy", "93.24%", help="SVM model accuracy on test set")
        
        with stat_col3:
            st.metric("Training Samples", "8,800", help="Dataset size used for training")
        
        with stat_col4:
            st.metric("Features Used", "7", help="Soil & climate parameters")
        
        st.markdown("---")
        
        # Sample use case
        st.subheader("💡 Sample Use Case")
        
        st.info("""
        **Makueni County Farmer:**
        
        *Inputs:*
        - N: 35 kg/ha (low, semi-arid soil)
        - Rainfall: 60mm (drought-prone)
        - Temperature: 28°C (hot)
        
        *Recommendation:* **CHICKPEA** (95% confidence)
        
        *Why it works:*
        - Low nitrogen needs (nitrogen-fixing legume)
        - Drought-tolerant (survives on 60mm rainfall)
        - Improves soil naturally (adds 40-80 kg N/ha)
        - Short harvest time (3 months = quick income)
        - Affordable ($40 total input cost)
        
        *Result:* Farmer plants chickpea, harvests 1.2 tons/ha, earns $900
        """)
        
        st.markdown("---")
        
        # Model insights
        st.subheader("🔬 How It Works")
        
        st.write("""
        The Intelligent Crop Recommendation System uses a Support Vector Machine (SVM) 
        classifier trained on 8,800 agricultural samples from diverse regions.
        
        **Key Features:**
        - **Multi-class classification**: Predicts from 22 different crop types
        - **Feature scaling**: StandardScaler normalizes all inputs for optimal performance
        - **Confidence scoring**: Provides probability-based confidence levels
        - **Out-of-distribution detection**: Warns about unusual input parameters
        - **Resource filtering**: Matches recommendations to farmer resources
        - **Climate scenario analysis**: Evaluates suitability under various conditions
        """)
        
        st.markdown("---")
        
        # Feature importance explanation
        st.subheader("📊 Which Factors Matter Most?")
        
        feature_importance = pd.DataFrame({
            'Feature': ['Nitrogen (N)', 'Potassium (K)', 'Phosphorus (P)', 
                       'Rainfall', 'Temperature', 'Humidity', 'pH'],
            'Importance (%)': [24.3, 18.7, 16.9, 15.2, 12.4, 7.8, 4.7]
        })
        
        fig_importance = px.bar(
            feature_importance,
            x='Importance (%)',
            y='Feature',
            orientation='h',
            color='Importance (%)',
            color_continuous_scale='Greens',
            title='Feature Importance in Crop Selection'
        )
        fig_importance.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.info("""
        **Key Insight:** NPK nutrients account for 60% of the model's decision-making.
        This validates the importance of soil testing for farmers.
        """)
        
        st.markdown("---")
        
        # Model reliability
        st.subheader("🎯 Model Reliability")
        
        reliability_data = pd.DataFrame({
            'Confidence Level': ['>95%', '85-95%', '70-85%', '<70%'],
            'Predictions': [1361, 167, 46, 186],
            'Accuracy (%)': [99.93, 99.40, 86.96, 62.0]
        })
        
        fig_reliability = px.bar(
            reliability_data,
            x='Confidence Level',
            y='Predictions',
            color='Accuracy (%)',
            color_continuous_scale='RdYlGn',
            title='Model Confidence vs Accuracy',
            text='Accuracy (%)'
        )
        fig_reliability.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_reliability.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_reliability, use_container_width=True)
        
        st.success("""
        **85% Confidence Threshold:** 
        - Covers 87% of all predictions
        - Achieves 99.4% accuracy
        - Predictions below 85% are flagged for review
        """)
        
        st.markdown("---")
        
        # Limitations
        with st.expander("⚠️ Limitations & Disclaimer"):
            st.warning("""
            **Model Limitations:**
            
            1. **Generalization:** Model trained on global data. May require regional calibration.
            
            2. **Feature Constraints:** Uses 7 parameters only. Doesn't account for:
               - Soil texture (clay/loam/sand)
               - Market prices (real-time)
               - Pest prevalence (region-specific)
               - Infrastructure access
            
            3. **Known Issues:**
               - 87% of errors involve groundnuts ↔ mothbeans confusion
               - Both have similar NPK profiles
               - Confidence threshold (85%) helps mitigate this
            
            4. **Resource Assumptions:** Assumes farmers can access:
               - Recommended seeds
               - Required fertilizers
               - Water (if irrigation needed)
               - Labor resources
            
            **Validation Status:**
            - Test accuracy: 93.24%
            - Cross-validation: 94.85% ± 0.21%
            - Field validation: Pending (pilot study planned)
            
            **Disclaimer:**
            This is a decision support tool, not a replacement for agricultural expertise.
            Always consult with local extension officers before making significant farming decisions.
            """)
        
        st.markdown("---")
        
        # About the project
        st.subheader("ℹ️ About This Project")
        
        st.write("""
        This Intelligent Crop Recommendation System was developed to support 
        **UN Sustainable Development Goal 2: Zero Hunger** by helping farmers make 
        data-driven decisions about crop selection.
        
        **Benefits:**
        - 🌾 Increased crop yields through optimal crop-soil matching
        - 💰 Reduced input waste and costs
        - 🌍 Promotion of sustainable agricultural practices
        - 📊 Data-driven decision making for smallholder farmers
        
        **Technology Stack:**
        - Machine Learning: Scikit-learn (SVM)
        - Web Framework: Streamlit
        - Data Processing: Pandas, NumPy
        - Visualization: Plotly
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌾 <strong>Intelligent Crop Recommendation System</strong> | Developed for SDG 2: Zero Hunger</p>
        <p>Helping farmers make data-driven decisions for sustainable agriculture</p>
        <p style='font-size: 0.8em; margin-top: 10px;'>
            <strong>Important:</strong> This system provides data-driven recommendations. 
            Always consult local agricultural experts before making final planting decisions.
            Regional validation recommended before large-scale deployment.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()