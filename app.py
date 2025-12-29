import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stroke Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Define risk categorization function
def categorize_stroke_risk(probability, thresholds=None):
    """Categorize stroke risk based on probability thresholds"""
    if thresholds is None:
        thresholds = {'Low': 0.3, 'Medium': 0.7}
    
    if probability < thresholds['Low']:
        return 'Low'
    elif probability < thresholds['Medium']:
        return 'Medium'
    else:
        return 'High'

def get_risk_description(probability):
    """Get user-friendly risk description instead of raw probability"""
    if probability < 0.3:
        return "Low Risk", "Low likelihood of stroke"
    elif probability < 0.5:
        return "Moderate Risk", "Some risk factors present"
    elif probability < 0.7:
        return "High Risk", "Significant stroke risk"
    elif probability < 0.9:
        return "High Risk", "Immediate attention recommended"
    else:
        return "High Risk", "Immediate attention recommended"

def get_key_risk_factors(patient_data):
    """Identify key risk factors from patient data"""
    risk_factors = []
    
    if patient_data.get('hypertension') == 'Yes':
        risk_factors.append('Hypertension')
    if patient_data.get('heart_disease') == 'Yes':
        risk_factors.append('Heart disease')
    if patient_data.get('smoking_status') == 'smokes':
        risk_factors.append('Smoking')
    if patient_data.get('age', 0) > 60:
        risk_factors.append('Age')
    if patient_data.get('avg_glucose_level', 0) > 140:
        risk_factors.append('High glucose')
    if patient_data.get('bmi', 0) > 30:
        risk_factors.append('Obesity')
    
    if risk_factors:
        return f"Based on risk factors: {' + '.join(risk_factors)}"
    else:
        return "Based on overall health profile"

# Define the pipeline class
class ImprovedStrokePredictionPipeline:
    def __init__(self, model, preprocessor=None, optimal_threshold=0.5, feature_names=None):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names if feature_names is not None else []
        self.optimal_threshold = optimal_threshold
        
    def _preprocess_patient_data(self, patient_data):
        """Preprocess patient data to match training data format"""
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Handle missing values
        if 'bmi' in patient_df.columns:
            patient_df['bmi'].fillna(patient_df['bmi'].median(), inplace=True)
        if 'smoking_status' in patient_df.columns:
            patient_df['smoking_status'].fillna('Unknown', inplace=True)
        
        # Remove id column if present
        if 'id' in patient_df.columns:
            patient_df = patient_df.drop('id', axis=1)
        
        # Encode categorical variables
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        
        for col in categorical_columns:
            if col in patient_df.columns:
                if col in ['ever_married', 'Residence_type']:
                    le = LabelEncoder()
                    patient_df[col] = le.fit_transform(patient_df[col].astype(str))
                else:
                    dummies = pd.get_dummies(patient_df[col], prefix=col)
                    patient_df = pd.concat([patient_df, dummies], axis=1)
                    patient_df = patient_df.drop(col, axis=1)
        
        # Create derived features
        if 'age' in patient_df.columns:
            patient_df['age_group'] = pd.cut(patient_df['age'], 
                                           bins=[0, 30, 50, 70, 100], 
                                           labels=['Young', 'Middle', 'Senior', 'Elderly'])
            patient_df['age_group'] = LabelEncoder().fit_transform(patient_df['age_group'].astype(str))
        
        if 'bmi' in patient_df.columns:
            patient_df['bmi_category'] = pd.cut(patient_df['bmi'], 
                                              bins=[0, 18.5, 25, 30, 100], 
                                              labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            patient_df['bmi_category'] = LabelEncoder().fit_transform(patient_df['bmi_category'].astype(str))
        
        if 'avg_glucose_level' in patient_df.columns:
            patient_df['glucose_category'] = pd.cut(patient_df['avg_glucose_level'], 
                                                  bins=[0, 100, 125, 200, 1000], 
                                                  labels=['Normal', 'Pre-diabetic', 'Diabetic', 'High'])
            patient_df['glucose_category'] = LabelEncoder().fit_transform(patient_df['glucose_category'].astype(str))
        
        # Create interaction terms
        if 'age' in patient_df.columns and 'bmi' in patient_df.columns:
            patient_df['age_bmi_interaction'] = patient_df['age'] * patient_df['bmi']
        
        if 'hypertension' in patient_df.columns and 'heart_disease' in patient_df.columns:
            patient_df['hypertension_heart_interaction'] = patient_df['hypertension'] * patient_df['heart_disease']
        
        # Create risk score
        risk_factors = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        patient_df['risk_score'] = sum(patient_df[col] for col in risk_factors if col in patient_df.columns)
        
        return patient_df
        
    def predict_stroke_risk(self, patient_data):
        """Predict stroke risk with detailed interpretation"""
        patient_processed = self._preprocess_patient_data(patient_data)
        
        # Ensure we have the same features as training data
        for feature in self.feature_names:
            if feature not in patient_processed.columns:
                patient_processed[feature] = 0
        
        # Reorder columns to match training data
        patient_processed = patient_processed[self.feature_names]
        
        # Get prediction using optimal threshold and probability
        probability = self.model.predict_proba(patient_processed)[0][1]
        
        # Validate probability is reasonable (medical models shouldn't be 100% certain)
        if probability >= 0.99:
            probability = 0.99  # Cap at 99% to avoid unrealistic certainty
            print(f"Warning: Probability capped at 99% for medical realism")
        elif probability <= 0.01:
            probability = 0.01  # Cap at 1% minimum
            
        prediction = 1 if probability >= self.optimal_threshold else 0
        
        # Categorize risk
        risk_category = categorize_stroke_risk(probability)
        
        # Get feature importance for interpretation
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = None
            
        # Create interpretation
        interpretation = {
            'prediction': 'Stroke Risk' if prediction == 1 else 'No Stroke Risk',
            'probability': probability,
            'risk_category': risk_category,
            'confidence': 'High' if probability > 0.8 or probability < 0.2 else 'Medium',
            'threshold_used': self.optimal_threshold
        }
        
        if importances is not None:
            # Get top 3 most important features
            top_features_idx = np.argsort(importances)[-3:][::-1]
            top_features = [(self.feature_names[i], importances[i]) for i in top_features_idx]
            interpretation['top_risk_factors'] = top_features
        
        return interpretation

# Also define the old class name for backward compatibility
StrokePredictionPipeline = ImprovedStrokePredictionPipeline

# Load the trained model and pipeline
@st.cache_data
def load_model():
    """Load the trained model and pipeline"""
    try:
        # Try to load the pipeline first
        pipeline = joblib.load('stroke_prediction_pipeline.pkl')
        
        # Ensure the pipeline has the optimal_threshold attribute
        if not hasattr(pipeline, 'optimal_threshold'):
            pipeline.optimal_threshold = 0.1  # Optimal medical threshold from optimization
        
        # Ensure the pipeline has feature_names
        if not hasattr(pipeline, 'feature_names') or not pipeline.feature_names:
            # Set default feature names based on your training data
            pipeline.feature_names = [
                'age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type',
                'avg_glucose_level', 'bmi', 'gender_Female', 'gender_Male', 'gender_Other',
                'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
                'work_type_children', 'work_type_selfemployed', 'smoking_status_Unknown',
                'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes',
                'age_group', 'bmi_category', 'glucose_category', 'risk_score',
                'age_bmi_interaction', 'hypertension_heart_interaction'
            ]
        
        return pipeline
        
    except Exception as e:
        # If pipeline loading fails, try loading just the model
        try:
            model = joblib.load('best_stroke_model.pkl')
            
            # Create a new pipeline with the loaded model
            feature_names = [
                'age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type',
                'avg_glucose_level', 'bmi', 'gender_Female', 'gender_Male', 'gender_Other',
                'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
                'work_type_children', 'work_type_selfemployed', 'smoking_status_Unknown',
                'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes',
                'age_group', 'bmi_category', 'glucose_category', 'risk_score',
                'age_bmi_interaction', 'hypertension_heart_interaction'
            ]
            
            pipeline = ImprovedStrokePredictionPipeline(
                model=model,
                optimal_threshold=0.1,
                feature_names=feature_names
            )
            
            return pipeline
            
        except Exception as e2:
            st.error(f"Error loading model: {str(e2)}")
            st.info("Make sure you have run the complete Jupyter notebook and saved the model files.")
            return None

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Stroke Risk Assessment Tool</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    pipeline = load_model()
    if pipeline is None:
        st.stop()
    
    # Sidebar for patient information
    st.sidebar.header("📋 Patient Information")
    
    # Basic Information
    st.sidebar.subheader("Basic Information")
    age = st.sidebar.slider("Age", min_value=1, max_value=100, value=50, help="Patient's age in years")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"], help="Patient's gender")
    
    # Medical History
    st.sidebar.subheader("Medical History")
    hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"], help="High blood pressure")
    heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"], help="History of heart disease")
    
    # Lifestyle Factors
    st.sidebar.subheader("Lifestyle Factors")
    ever_married = st.sidebar.selectbox("Ever Married", ["No", "Yes"], help="Marital status")
    work_type = st.sidebar.selectbox("Work Type", 
                                   ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                                   help="Type of work")
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"], help="Type of residence")
    smoking_status = st.sidebar.selectbox("Smoking Status", 
                                        ["never smoked", "formerly smoked", "smokes"],
                                        help="Current smoking status")
    
    # Health Metrics
    st.sidebar.subheader("Health Metrics")
    avg_glucose_level = st.sidebar.number_input("Average Glucose Level", 
                                              min_value=50.0, max_value=300.0, 
                                              value=100.0, step=1.0,
                                              help="Average glucose level in mg/dL")
    bmi = st.sidebar.number_input("BMI (Body Mass Index)", 
                                min_value=10.0, max_value=60.0, 
                                value=25.0, step=0.1,
                                help="Body Mass Index")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Risk Assessment")
        
        # Create patient data dictionary
        patient_data = {
            'age': age,
            'gender': gender,
            'hypertension': 1 if hypertension == "Yes" else 0,
            'heart_disease': 1 if heart_disease == "Yes" else 0,
            'ever_married': 1 if ever_married == "Yes" else 0,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }
        
        # Display patient summary
        st.subheader("Patient Summary")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Age", f"{age} years")
            st.metric("Gender", gender)
            st.metric("Hypertension", hypertension)
        
        with col_b:
            st.metric("Heart Disease", heart_disease)
            st.metric("Work Type", work_type)
            st.metric("Residence", residence_type)
        
        with col_c:
            st.metric("BMI", f"{bmi:.1f}")
            st.metric("Glucose Level", f"{avg_glucose_level:.1f} mg/dL")
            st.metric("Smoking", smoking_status)
        
        # Predict button
        if st.button("🔬 Assess Stroke Risk", type="primary", use_container_width=True):
            try:
                # Make prediction
                result = pipeline.predict_stroke_risk(patient_data)
                
                # Display results
                st.subheader("📊 Risk Assessment Results")
                
                # Risk probability
                probability = result['probability']
                risk_category = result['risk_category']
                
                # Get user-friendly risk description and risk factors
                risk_level, risk_description = get_risk_description(probability)
                risk_factors_text = get_key_risk_factors(patient_data)
                
                # Display risk category with appropriate styling
                if risk_category == 'Low':
                    st.markdown(f'<div class="risk-low"><h3>🟢 {risk_level}</h3><p>{risk_description}</p><p><small>{risk_factors_text}</small></p></div>', 
                              unsafe_allow_html=True)
                elif risk_category == 'Medium':
                    st.markdown(f'<div class="risk-medium"><h3>🟡 {risk_level}</h3><p>{risk_description}</p><p><small>{risk_factors_text}</small></p></div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-high"><h3>🔴 {risk_level}</h3><p>{risk_description}</p><p><small>{risk_factors_text}</small></p></div>', 
                              unsafe_allow_html=True)
                
                # Detailed metrics
                col1_metrics, col2_metrics = st.columns(2)
                
                with col1_metrics:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Risk Level", risk_level)
                    st.metric("Risk Category", risk_category)
                    st.metric("Confidence", result['confidence'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2_metrics:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Prediction", result['prediction'])
                    st.metric("Threshold Used", f"{result['threshold_used']:.2f}")
                    st.metric("Model Type", "Machine Learning")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Top risk factors
                if 'top_risk_factors' in result:
                    st.subheader("🎯 Top Risk Factors")
                    for i, (feature, importance) in enumerate(result['top_risk_factors'], 1):
                        st.write(f"{i}. **{feature.replace('_', ' ').title()}**: {importance:.3f}")
                
                # Recommendations
                st.subheader("💡 Medical Recommendations")
                if risk_category == 'Low':
                    st.success("✅ **Continue healthy lifestyle** - Maintain current habits and regular check-ups.")
                elif risk_category == 'Medium':
                    st.warning("⚠️ **Monitor closely** - Consider lifestyle changes and regular monitoring.")
                    st.write("- Regular blood pressure monitoring")
                    st.write("- Healthy diet and exercise")
                    st.write("- Regular medical check-ups")
                else:
                    st.error("🚨 **Consult healthcare provider immediately** - High risk requires medical attention.")
                    st.write("- Immediate medical consultation")
                    st.write("- Comprehensive health assessment")
                    st.write("- Consider preventive medications")
                    st.write("- Lifestyle modifications")
                
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                st.write("Please check the model files and try again.")
    
    with col2:
        st.header("ℹ️ About This Tool")
        
        st.info("""
        **Stroke Risk Assessment Tool**
        
        This application uses machine learning to assess stroke risk based on patient characteristics and medical history.
        
        **Risk Categories:**
        - 🟢 **Low Risk**: < 30% probability
        - 🟡 **Medium Risk**: 30-70% probability  
        - 🔴 **High Risk**: > 70% probability
        
        **Disclaimer:**
        This tool is for educational purposes only and should not replace professional medical advice.
        """)
        
        st.subheader("📈 Risk Factors")
        st.write("""
        **Modifiable Risk Factors:**
        - High blood pressure
        - Smoking
        - High cholesterol
        - Diabetes
        - Physical inactivity
        - Obesity
        
        **Non-modifiable Risk Factors:**
        - Age
        - Gender
        - Family history
        - Previous stroke
        """)
        
        st.subheader("🛡️ Prevention Tips")
        st.write("""
        1. **Control blood pressure**
        2. **Quit smoking**
        3. **Manage diabetes**
        4. **Exercise regularly**
        5. **Eat a healthy diet**
        6. **Limit alcohol**
        7. **Manage stress**
        """)

if __name__ == "__main__":
    main()
