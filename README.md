# Crop Recommendation System using Machine Learning

**Python 3.8+** | **scikit-learn** | **XGBoost** | **Power BI** | **License: MIT**

---

## 🌾 Project Overview

This project aims to develop a machine learning-based crop recommendation system that provides data-driven agricultural guidance to farmers based on soil composition and climatic conditions. Using the comprehensive Crop Recommendation Dataset, we will build and compare multiple classification models to identify optimal crop selections, helping farmers maximize yields while reducing resource wastage.

### Agricultural Impact Potential
- **Target yield increase**: 15-25% through optimal crop selection
- **Resource optimization**: 30% reduction in fertilizer waste through precise soil-crop matching
- **Global relevance**: Support for smallholder farmers facing crop selection challenges
- **Sustainability**: Contribution to SDG 2: Zero Hunger through improved food security

---

## 📊 Dataset

### Primary Dataset: Crop Recommendation Dataset
- **Scale**: 8800 samples across 22 crop types
- **Features**: 7 input parameters (soil nutrients and climate conditions)
- **Quality**: Clean dataset with no missing values

### Feature Description

| Feature | Description | Unit | Agricultural Significance |
|---------|-------------|------|--------------------------|
| **N** | Nitrogen content | kg/ha | Critical for leaf growth and chlorophyll |
| **P** | Phosphorus content | kg/ha | Essential for root development |
| **K** | Potassium content | kg/ha | Regulates water uptake and enzyme activation |
| **temperature** | Average temperature | °C | Determines climate suitability |
| **humidity** | Relative humidity | % | Affects transpiration and disease pressure |
| **ph** | Soil pH value | - | Controls nutrient availability |
| **rainfall** | Annual rainfall | mm | Determines water requirements |

### Target Crops (22 Types)

Cereals: rice, maize
Pulses: chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil
Fruits: pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut
Cash Crops: cotton, jute, coffee


---

## 🏗️ Technical Architecture

### Machine Learning Approaches

#### 1. **Random Forest Classifier** (Primary Model)

#### 2. **XGBoost Classifier** (Secondary Model)

#### 3. **Support Vector Machine** (Baseline)

#### 4. **K-Nearest Neighbors** (Baseline)

### Tech Stack

```python
# Core Machine Learning
scikit-learn, xgboost, pandas, numpy

# Visualization & Analysis
matplotlib, seaborn, plotly, Power BI

# Web Interface & Deployment
streamlit, flask, sqlite3, docker

# Development & Testing
jupyter, pytest, black, flake8
```

### Project Structure

crop-recommendation-system/
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code modules
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # ML model implementations
│   ├── training/          # Training pipelines
│   ├── evaluation/        # Model evaluation metrics
│   └── utils/             # Configuration and logging
├── app/                   # Web application and API
├── sql_queries/           # Database schema and queries
├── powerbi/               # Business intelligence dashboard
├── models/                # Saved trained models
├── tests/                 # Unit and integration tests
└── docs/                  # Project documentation

## 🚀 Implementation Plan

### Phase 1: Data Preparation & Exploration
- Load and clean dataset
- Perform exploratory data analysis (EDA)
- Create SQLite database with analytical queries
- Generate initial visualizations and insights

### Phase 2: Model Development
- Implement and train 4 classification models
- Perform hyperparameter tuning and cross-validation
- Evaluate model performance using multiple metrics
- Select best-performing model for deployment

### Phase 3: Business Intelligence
- Develop interactive Power BI dashboard
- Create 5+ meaningful visualizations
- Implement DAX measures for agricultural insights
- Design user-friendly interface for data exploration

### Phase 4: Application Development
- Build Streamlit web application for real-time predictions
- Develop Flask REST API for integration capabilities
- Create responsive and intuitive user interface
- Implement model inference pipeline

### Phase 5: Deployment & Documentation
- Containerize application using Docker
- Deploy web application for public access
- Create comprehensive documentation
- Prepare project portfolio materials

---

## 🎯 Success Metrics

### Technical Performance
- **Model Accuracy**: Target ≥95% on test dataset
- **API Response Time**: <500ms per prediction
- **Dashboard Load Time**: <5 seconds
- **Code Coverage**: ≥80% test coverage

### Agricultural Impact
- **Usability**: Accessible to non-technical users
- **Actionability**: Clear, interpretable recommendations
- **Scalability**: Support for multiple regions and crop types
- **Reliability**: Consistent performance across diverse conditions

### Project Deliverables
1. **Trained ML Models**: 4 pickled models with performance documentation
2. **SQL Database**: Structured database with analytical queries
3. **Power BI Dashboard**: Interactive agricultural insights platform
4. **Web Application**: User-friendly crop recommendation interface
5. **Documentation**: Comprehensive project documentation and guides

---

## 🔬 Expected Insights

### Agricultural Knowledge Discovery
- **Nutrient-Crop Relationships**: Identify optimal NPK ranges for each crop
- **Climate Adaptation**: Map crops to temperature and rainfall patterns
- **Soil Health**: Recommendations for pH optimization and soil management
- **Resource Efficiency**: Strategies for water and fertilizer optimization

### Business Intelligence
- **Crop Suitability Analysis**: Regional adaptation recommendations
- **Economic Optimization**: High-value crop identification
- **Risk Mitigation**: Alternative crop suggestions for climate resilience
- **Sustainability Metrics**: Environmental impact assessment

---

## 🌍 Sustainable Development Alignment

This project directly supports **UN Sustainable Development Goal 2: Zero Hunger** by:
- Improving agricultural productivity through data-driven decisions
- Reducing resource waste in farming practices
- Enhancing food security through optimal crop selection
- Supporting smallholder farmers with accessible technology

---

## 🤝 Contributing

We welcome contributions from:
- **Data Scientists**: Model improvements and feature engineering
- **Agronomists**: Domain expertise and validation
- **Developers**: Web interface enhancements and API development
- **Designers**: User experience improvements and visualization
- **Farmers**: Real-world testing and feedback

### Contribution Areas
- Expand crop database with regional varieties
- Integrate real-time weather data APIs
- Develop mobile application interfaces
- Add multi-language support
- Implement advanced features like crop rotation planning

---

## 📞 Contact & Links

**Project Maintainer**: Chelah Ng'etich  
**Email**: ngetichchelah@gmail.com  
**GitHub**: [github.com/ngetichchelah](https://github.com/ngetichchelah)  
**LinkedIn**: [linkedin.com/in/ngetich-chelangat](https://linkedin.com/in/ngetich-chelangat)

---

## 📋 Overall Steps Undertaken

1. **Environment Setup**: Install dependencies and configure development environment
2. **Data Acquisition**: Download and explore the Kaggle dataset
3. **Database Creation**: Implement SQLite database with analytical queries
4. **Model Development**: Begin training and evaluating machine learning models
5. **Dashboard Development**: Create Power BI visualizations and insights
6. **Application Building**: Develop web interface and API endpoints
7. **Testing & Deployment**: Validate system performance and deploy for access

---

*This project represents a comprehensive approach to applying data science for agricultural improvement, combining technical excellence with real-world impact potential.* 🌱
