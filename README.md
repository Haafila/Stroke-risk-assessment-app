# Stroke Risk Assessment - Complete Package

A complete machine learning pipeline for stroke risk prediction, integrated with a professional Streamlit web application. This project was developed as a Group Mini Project for the 3rd Year – 1st Semester, Fundamentals of Data Mining module.
## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Open Your Browser
Go to `http://localhost:8501`

## 📊 Features

### **Machine Learning Pipeline:**
- ✅ **Complete EDA** - Exploratory data analysis with visualizations
- ✅ **Data Preprocessing** - Missing value handling, encoding, scaling
- ✅ **Feature Engineering** - Age groups, BMI categories, interaction terms
- ✅ **Model Training** - Multiple algorithms with cross-validation
- ✅ **Model Evaluation** - Comprehensive metrics and visualizations
- ✅ **Threshold Optimization** - Medical-grade sensitivity tuning

### **Web Application:**
- ✅ **Interactive Interface** - Professional medical UI
- ✅ **Real-time Predictions** - Instant stroke risk assessment
- ✅ **Risk Categorization** - Low/Medium/High risk levels
- ✅ **Feature Importance** - Top risk factors analysis
- ✅ **Medical Recommendations** - Actionable health advice
- ✅ **Responsive Design** - Works on desktop and mobile

## 🏥 Medical Features

### **Risk Categories:**
- 🟢 **Low Risk**: < 30% probability
- 🟡 **Medium Risk**: 30-70% probability
- 🔴 **High Risk**: > 70% probability

### **Risk Factors Analyzed:**
- Age (most important factor)
- Hypertension
- Heart disease
- Smoking status
- BMI categories
- Glucose levels
- Gender differences
- Lifestyle factors

### **Model Performance:**
- **Algorithm**: Logistic Regression with balanced class weights
- **ROC-AUC**: ~0.83 (realistic and trustworthy)
- **Threshold**: 0.3 (optimized for medical sensitivity)
- **Features**: 16 engineered features

## 🔧 Technical Details

### **Model Architecture:**
- **Preprocessing**: Missing value imputation, categorical encoding
- **Feature Engineering**: Derived features, interaction terms
- **Model**: Logistic Regression with class balancing
- **Validation**: Stratified k-fold cross-validation
- **Deployment**: Scikit-learn Pipeline with joblib serialization

### **Data Pipeline:**
1. **Data Loading** - CSV import with validation
2. **EDA** - Comprehensive exploratory analysis
3. **Preprocessing** - Systematic data cleaning
4. **Feature Engineering** - Domain-specific feature creation
5. **Model Training** - Multiple algorithm comparison
6. **Evaluation** - Medical-grade performance metrics
7. **Deployment** - Production-ready pipeline

## 🛡️ Medical Disclaimer

This tool is for educational and research purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.


