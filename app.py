import os
import sys
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables (ignore if .env doesn't exist)
load_dotenv(verbose=False)

# Constants
MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'model.joblib'
DATA_PATH = Path('diabetes.csv')
RANDOM_STATE = 42  # For reproducible results

def setup_environment():
    """Set up the application environment."""
    # Ensure models directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Environment setup complete")

def load_data():
    """Load and return the diabetes dataset."""
    try:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Successfully loaded data from {DATA_PATH}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """Preprocess the input DataFrame."""
    try:
        # Remove SkinThickness outlier
        max_skinthickness = df.SkinThickness.max()
        df = df[df.SkinThickness != max_skinthickness]
        
        # Replace zeros with mean by class for relevant columns
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            df = replace_zero(df, col, 'Outcome')
        
        logger.info("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

def replace_zero(df, field, target):
    """Replace zeros with mean values grouped by the target variable."""
    try:
        mean_by_target = df.loc[df[field] != 0, [field, target]].groupby(target).mean()
        df.loc[(df[field] == 0) & (df[target] == 0), field] = mean_by_target.iloc[0][0]
        df.loc[(df[field] == 0) & (df[target] == 1), field] = mean_by_target.iloc[1][0]
        return df
    except Exception as e:
        logger.error(f"Error in replace_zero for {field}: {e}")
        raise

def train_model(X, y):
    """Train and return the Gradient Boosting model."""
    try:
        # Gradient Boosting with optimized parameters
        gb_params = {
            'n_estimators': 290,
            'max_depth': 9,
            'subsample': 0.5,
            'learning_rate': 0.01,
            'min_samples_leaf': 1,
            'random_state': RANDOM_STATE
        }
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(**gb_params))
        ])
        
        # Train the model
        pipeline.fit(X, y)
        logger.info("Model training completed successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model):
    """Save the trained model to disk."""
    try:
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load_saved_model():
    """Load a previously saved model from disk."""
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info("Loaded saved model")
            return model
        return None
    except Exception as e:
        logger.error(f"Error loading saved model: {e}")
        return None

def get_user_input():
    """Get input features from the user via the sidebar."""
    st.sidebar.header('Patient Information')
    
    # Group related features
    with st.sidebar.expander("Personal Information"):
        pregnancies = st.slider('Pregnancies', 0, 17, 3)
        age = st.slider('Age', 21, 81, 29)
    
    with st.sidebar.expander("Medical Measurements"):
        col1, col2 = st.columns(2)
        with col1:
            glucose = st.slider('Glucose (mg/dL)', 0, 200, 120)
            bp = st.slider('Blood Pressure (mmHg)', 0, 122, 70)
            skin_thickness = st.slider('Skin Thickness (mm)', 0, 100, 20)
        with col2:
            insulin = st.slider('Insulin (μU/mL)', 0, 846, 79)
            bmi = st.slider('BMI', 0.0, 67.1, 32.0)
            dpf = st.slider('Diabetes Pedigree', 0.0, 2.42, 0.3725, 0.01)
    
    # Create feature dictionary
    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    return pd.DataFrame(user_data, index=[0])

def main():
    """Main application function."""
    try:
        # Set page config
        st.set_page_config(
            page_title="Diabetes Prediction App",
            page_icon="🩺",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Setup environment
        setup_environment()
        
        # Load and preprocess data
        with st.spinner('Loading and preprocessing data...'):
            df = load_data()
            df = preprocess_data(df)
            X = df.drop(['Outcome'], axis=1)
            y = df['Outcome']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )
        
        # Check for saved model, otherwise train a new one
        model = load_saved_model()
        if model is None:
            with st.spinner('Training model...'):
                model = train_model(X_train, y_train)
                save_model(model)
        
        # Main app layout
        st.title('Diabetes Risk Assessment')
        st.markdown("""
        This application uses machine learning to predict the likelihood of diabetes 
        based on patient health metrics. Enter the patient's information in the sidebar 
        and click 'Predict' to see the results.
        """)
        
        # Get user input and display it
        try:
            user_data = get_user_input()
            
            # Display user input in main area
            with st.expander("View Patient Data"):
                st.dataframe(user_data.style.format({
                    'Pregnancies': '{:.0f}',
                    'Glucose': '{:.0f} mg/dL',
                    'BloodPressure': '{:.0f} mm Hg',
                    'SkinThickness': '{:.0f} mm',
                    'Insulin': '{:.0f} mu U/ml',
                    'BMI': '{:.1f} kg/m²',
                    'DiabetesPedigreeFunction': '{:.3f}',
                    'Age': '{:.0f} years'
                }))
                
                # Make prediction when user clicks the button
                if st.button('Predict Diabetes Risk'):
                    with st.spinner('Analyzing...'):
                        prediction = model.predict(user_data)
                        prediction_proba = model.predict_proba(user_data)
                        
                        # Display results
                        st.subheader('Prediction Results')
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Risk of Diabetes",
                                "High Risk" if prediction[0] == 1 else "Low Risk",
                                f"{prediction_proba[0][1] * 100:.2f}%"
                            )
                        with col2:
                            st.metric(
                                "Confidence",
                                f"{np.max(prediction_proba) * 100:.1f}%",
                                "in prediction"
                            )
                            
                        # Display feature importance
                        st.subheader('Feature Importance')
                        if hasattr(model, 'feature_importances_'):
                            feature_importance = pd.DataFrame({
                                'Feature': X_train.columns,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(
                                x='Importance', 
                                y='Feature', 
                                data=feature_importance,
                                palette='viridis'
                            )
                            plt.title('Feature Importance')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main app: {str(e)}", exc_info=True)
        # Make prediction
        if st.sidebar.button('Predict', type='primary'):
            with st.spinner('Analyzing...'):
                # Make prediction
                prediction = model.predict(user_data)
                proba = model.predict_proba(user_data)[0]
                
                # Display prediction with emoji and styling
                st.markdown("## Prediction Result")
                if prediction[0] == 1:
                    st.error('⚠️ **High Risk of Diabetes**')
                    st.warning("""
                    Based on the provided information, this patient shows indicators associated 
                    with a higher risk of diabetes. We recommend consulting with a healthcare 
                    professional for further evaluation.
                    """)
                else:
                    st.success('✅ **Low Risk of Diabetes**')
                    st.info("""
                    Based on the provided information, this patient shows indicators associated 
                    with a lower risk of diabetes. However, regular check-ups are recommended 
                    for maintaining good health.
                    """)
                
                # Show probability with a progress bar
                risk_percent = proba[1] * 100
                st.metric("Risk Score", f"{risk_percent:.1f}%")
                st.progress(risk_percent / 100)
                
                # Model performance
                st.markdown("## Model Performance")
                
                # Cross-validated metrics
                with st.expander("Performance Metrics"):
                    col1, col2 = st.columns(2)
                    
                    # Cross-validated F1
                    with col1:
                        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
                        st.metric(
                            "Cross-validated F1-score",
                            f"{cv_scores.mean():.3f}",
                            f"±{cv_scores.std():.3f}"
                        )
                    
                    # Test set metrics
                    with col2:
                        y_pred = model.predict(X_test)
                        test_accuracy = accuracy_score(y_test, y_pred)
                        st.metric(
                            "Test Set Accuracy",
                            f"{test_accuracy*100:.1f}%"
                        )
                
                # Feature importance
                st.markdown("## Feature Importance")
                importances = model.named_steps['classifier'].feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                # Horizontal bar chart for feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feature_importance['Feature'], feature_importance['Importance'])
                ax.set_xlabel('Importance Score')
                ax.set_title('Relative Importance of Features in Prediction')
                st.pyplot(fig)
                
                # Confusion matrix
                st.markdown("## Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    ax=ax,
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes']
                )
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Model Performance on Test Data')
                st.pyplot(fig)
                
                # Add footer
                st.markdown("---")
                st.caption("""
                **Note**: This tool is for informational purposes only and is not intended 
                to replace professional medical advice, diagnosis, or treatment. Always seek 
                the advice of your physician or other qualified health provider with any 
                questions you may have regarding a medical condition.
                """)
    
    except FileNotFoundError:
        st.error("""
        ### Error: Data File Not Found
        The diabetes dataset could not be found. Please ensure that `diabetes.csv` 
        is in the project directory.
        """)
        logger.error("diabetes.csv file not found")
    except Exception as e:
        st.error(f"""
        ### An Unexpected Error Occurred
        We apologize for the inconvenience. The application encountered an error:
        
        `{str(e)}`
        
        Please try refreshing the page or contact support if the issue persists.
        """)
        logger.exception("Unexpected error in main application")

if __name__ == "__main__":
    main()