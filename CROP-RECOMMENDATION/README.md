# **Crop Recommendation System**

### **Project Objective**
To build a machine learning system that recommends **optimal crops** for cultivation based on **soil composition and climatic conditions**.  

This project demonstrates **end-to-end ML workflow skills**: data preprocessing, exploratory data analysis (EDA), feature engineering, multi-class classification modeling, model evaluation, deployment, and insights generation.  

**Purpose:** Assist farmers, agronomists, and agricultural planners in selecting crops that maximize yield and adapt to environmental conditions.

---

### **Dataset**
**Source:** [Crop Recommendation Dataset]  

**Features (independent variables):**
- `N` → Nitrogen content in soil  
- `P` → Phosphorous content in soil  
- `K` → Potassium content in soil  
- `temperature` → Average temperature in °C  
- `humidity` → Average humidity (%)  
- `ph` → Soil pH level  
- `rainfall` → Rainfall in mm  

**Target (dependent variable):**
- `label` → Recommended crop (22 classes)

---

### **Methodology / Process**

### ***1. Data Preprocessing***
- Handle missing or inconsistent values  
- Encode categorical variables if needed  
- Split dataset into training and test sets  
- Scale numerical features for model input  

### ***2. Exploratory Data Analysis (EDA)***
- Visualize nutrient distributions for each crop  
- Study correlations between features and target crop  
- Detect patterns in temperature, rainfall, and pH preferences  
- Identify outliers or extreme values that may skew predictions  

**Key Visualizations:**
1. Feature distributions per crop → highlights nutrient ranges  
2. Correlation heatmap → reveals interactions between N, P, K, temperature, humidity, pH, and rainfall  
3. Boxplots → show variation for each crop  

### ***3. Feature Engineering***
- Normalize nutrient, temperature, and rainfall data  
- Create interaction terms if needed (e.g., N × P, temperature × humidity)  
- Handle skewed or extreme values for robust input to models  

### ***4. Modeling Approaches***
#### Classical Machine Learning
- **Random Forest**: Non-linear tree-based ensemble, robust to feature scaling  
- **XGBoost / Gradient Boosting**: Handles non-linear relationships with feature importance analysis  
- **Support Vector Machine (SVM)**: Effective with smaller datasets and high-dimensional features  
- **K-Nearest Neighbors (KNN)**: Simple instance-based learning for comparison  

#### Ensemble Approach
- Weighted voting across Random Forest, XGBoost, and SVM for improved accuracy  

**Key Modeling Insights:**
- Random Forest often achieved **highest accuracy (>98%)**  
- XGBoost provided faster training with comparable accuracy  
- SVM and KNN performed well but were sensitive to feature scaling  

### ***5. Model Evaluation***
- Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix  
- Feature importance analysis for interpretability  
- Cross-validation to ensure generalization  

### ***6. Deployment / Output***
- Streamlit web app for farmer-friendly input of soil and climate features  
- Outputs recommended crop(s) with confidence scores  
- Optional alternative crop suggestions for crop rotation planning  

---

### **End Goals of This Project**
**Technical Deliverables**
- Train and compare 4 ML models (Random Forest, XGBoost, SVM, KNN)  
- Achieve 90%+ prediction accuracy on crop recommendations  
- Create SQLite database with 10+ analytical queries  
- Build interactive Power BI dashboard with 5+ visualizations  
- Develop web application (Streamlit) for real-time predictions  
- Deploy REST API (Flask) for integration capabilities  
- Generate comprehensive model performance reports  
- Document entire codebase with proper structure  

**Data Science Skills Demonstrated**
- End-to-end ML pipeline (data → model → deployment)  
- Multi-class classification problem solving  
- Feature importance analysis and model explainability  
- Cross-validation and hyperparameter tuning  
- Ensemble learning techniques  
- SQL database design and querying  
- Business intelligence dashboard creation  
- Model serialization and deployment  

**Business / Agricultural Impact**
- Provide data-driven crop recommendations to farmers  
- Reduce fertilizer waste through optimal crop selection  
- Increase crop yields by 15–25% via soil-crop matching  
- Enable smallholder farmers to make informed decisions  
- Contribute to SDG 2: Zero Hunger goals  
- Create tool accessible via web interface (no technical knowledge needed)  
- Generate actionable agricultural insights from soil data  

**Portfolio & Career Goals**
- Complete project showcasing Python, SQL, Power BI, ML skills  
- GitHub repository with professional documentation  
- Live deployed application (demo link for resume)  
- Case study demonstrating real-world problem-solving  
- Evidence of ability to work on SDG-aligned projects  
- Showcase data storytelling through visualizations  
- Demonstrate full-stack data science capabilities  

**Key Outputs**
1. Trained Models: 4 pickled ML models with good accuracy  
2. Database: SQLite with 2,200+ records and analytical queries  
3. Dashboard: Power BI file with interactive crop analysis  
4. Web App: Streamlit application with prediction interface  
5. API: Flask REST API for programmatic access  
6. Documentation: Comprehensive README and code comments  
7. Notebook: Jupyter notebooks showing entire analysis  
8. Reports: Performance metrics, confusion matrices, feature importance charts  

**Success Metrics**
- 📊 Model accuracy: ≥90%  
- 📊 API response time: <500ms  
- 📊 Dashboard load time: <5 seconds  
- 📊 Code coverage: ≥80%  
- 📊 GitHub stars/forks: Community engagement  
- 📊 User feedback: Positive reception from testing  

---

**Technology Stack**
- Python 3.8+  
- Libraries: NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib/Seaborn, Plotly, Joblib  
- Web Interface: Streamlit  
- API: Flask  
- Database: SQLite  
- BI Dashboard: Power BI  

graph TD
    A[Dataset: Soil Nutrients + Climate] --> B[Preprocessing & Feature Engineering]
    B --> C[Exploratory Data Analysis]
    C --> D[Baseline Models: KNN, SVM]
    C --> E[Advanced Models: Random Forest, XGBoost]
    E --> F[Model Evaluation & Comparison]
    F --> G[Insights & Recommendations]
    G --> H[Deployment: Streamlit Web App & Flask API]
    H --> I[Business Intelligence: Power BI Dashboard]



