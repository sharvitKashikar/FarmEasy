import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌱 Sustainable Agriculture Yield Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #32CD32);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #32CD32;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .tip-box {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #FF6B35;
    }
    .scheme-box {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4facfe;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_prepare_data():
    """Load and prepare the crop yield data"""
    try:
        df = pd.read_csv('crop_yield.csv')
        
        # Clean data
        df = df.dropna()
        
        # Remove extreme outliers (keep 95% of data)
        for col in ['Yield', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']:
            if col in df.columns:
                Q1 = df[col].quantile(0.025)
                Q3 = df[col].quantile(0.975)
                df = df[(df[col] >= Q1) & (df[col] <= Q3)]
        
        # Feature engineering
        df['Fertilizer_per_Area'] = df['Fertilizer'] / (df['Area'] + 1e-6)
        df['Pesticide_per_Area'] = df['Pesticide'] / (df['Area'] + 1e-6)
        df['Efficiency'] = df['Production'] / (df['Fertilizer'] + df['Pesticide'] + 1e-6)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_resource
def train_model(df):
    """Train a robust yield prediction model"""
    
    # Prepare features
    features = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 
               'Fertilizer_per_Area', 'Pesticide_per_Area']
    
    # Encode categorical variables
    le_crop = LabelEncoder()
    le_season = LabelEncoder()
    le_state = LabelEncoder()
    
    df['Crop_encoded'] = le_crop.fit_transform(df['Crop'])
    df['Season_encoded'] = le_season.fit_transform(df['Season'])
    df['State_encoded'] = le_state.fit_transform(df['State'])
    
    features.extend(['Crop_encoded', 'Season_encoded', 'State_encoded', 'Crop_Year'])
    
    X = df[features]
    y = df['Yield']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Cross validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    return {
        'model': rf_model,
        'scaler': scaler,
        'label_encoders': {'Crop': le_crop, 'Season': le_season, 'State': le_state},
        'features': features,
        'metrics': {
            'r2': r2, 'mae': mae, 'rmse': rmse,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
        },
        'test_data': {'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}
    }


def get_government_schemes(state, crop):
    schemes = [
        "🌾 **PMFBY** - Crop insurance scheme",
        "💰 **PM-KISAN** - ₹6,000 annual income support",
        "🚜 **Farm Mechanization Subsidies**",
        "🌱 **National Mission for Sustainable Agriculture**",
        "💧 **PMKSY - Irrigation Efficiency Improvement**"
    ]
    
    state_schemes = {
        'Punjab': ["🌾 Punjab Crop Diversification", "💧 Free Electricity for Agriculture"],
        'Uttar Pradesh': ["🌱 UP Crop Diversification Scheme"],
        'Maharashtra': ["☔ Drought Relief Scheme", "🍯 Honey Mission"],
        'West Bengal': ["🌾 Krishak Bandhu Scheme"]
    }
    
    if state in state_schemes:
        schemes.extend(state_schemes[state])
    
    return schemes


def get_yield_improvement_tips(prediction, crop, inputs):
    tips = []
    if prediction < 2:
        tips.append("🚨 Low Yield: Improve soil nutrition & organic content")
    elif prediction < 5:
        tips.append("📈 Moderate Yield: Good potential to improve")
    else:
        tips.append("🌟 Excellent Yield: Maintain current practices")
    
    fert_per_area = inputs['Fertilizer_per_Area']
    if fert_per_area < 20:
        tips.append("🌱 Increase fertilizer by 20-25%")
    elif fert_per_area > 100:
        tips.append("⚠️ Reduce fertilizer use by 15%")
    
    return tips


def create_yield_comparison_chart(prediction, crop):
    regional_data = {
        'Rice': {'Punjab': 6.2, 'West Bengal': 5.8, 'National': 4.2},
        'Wheat': {'Punjab': 5.1, 'UP': 3.2, 'National': 3.5}
    }
    
    if crop not in regional_data:
        return None
    
    data = regional_data[crop]
    regions = list(data.keys()) + ['Your Farm']
    values = list(data.values()) + [prediction]
    
    fig = px.bar(x=regions, y=values, title=f"{crop} Yield Comparison")
    return fig


def create_feature_impact_chart(model_package):
    importances = model_package['model'].feature_importances_
    features = [f.replace('_', ' ').title() for f in model_package['features']]
    
    fig = px.bar(x=importances, y=features, orientation='h',
                title='Feature Importance')
    return fig


def main():
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Sustainable Agriculture Yield Predictor</h1>
        <p>AI-powered crop yield prediction with insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_and_prepare_data()
    if df is None:
        return
    
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🔮 Yield Prediction", 
        "📊 Model Performance", 
        "📈 Analytics Dashboard"
    ])
    
    if page == "🔮 Yield Prediction":
        prediction_page(df)
    elif page == "📊 Model Performance":
        model_performance_page(df)
    elif page == "📈 Analytics Dashboard":
        analytics_page(df)


def prediction_page(df):
    st.header("🔮 Crop Yield Prediction")
    
    with st.spinner("Training model..."):
        model_package = train_model(df)
    
    st.success(f"Model Ready! R²: {model_package['metrics']['r2']:.3f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        crop = st.selectbox("Crop", sorted(df["Crop"].unique()))
        season = st.selectbox("Season", sorted(df["Season"].unique()))
        state = st.selectbox("State", sorted(df["State"].unique()))
        crop_year = st.number_input("Crop Year", 1997, 2030, 2024)
    
    with col2:
        area = st.number_input("Area (ha)", 1.0, value=100.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, value=800.0)
        fertilizer = st.number_input("Fertilizer (kg)", 0.0, value=5000.0)
        pesticide = st.number_input("Pesticide (kg)", 0.0, value=50.0)
    
    if st.button("🎯 Predict Yield"):
        input_data = {
            'Area': area,
            'Annual_Rainfall': rainfall,
            'Fertilizer': fertilizer,
            'Pesticide': pesticide,
            'Fertilizer_per_Area': fertilizer / area,
            'Pesticide_per_Area': pesticide / area,
            'Crop_encoded': model_package['label_encoders']['Crop'].transform([crop])[0],
            'Season_encoded': model_package['label_encoders']['Season'].transform([season])[0],
            'State_encoded': model_package['label_encoders']['State'].transform([state])[0],
            'Crop_Year': crop_year
        }
        
        input_df = pd.DataFrame([input_data])
        scaled_data = model_package['scaler'].transform(input_df[model_package['features']])
        prediction = model_package['model'].predict(scaled_data)[0]
        
        st.markdown(f"""
        <div class='prediction-box'>
            <h2>🎯 Predicted Yield</h2>
            <h1>{prediction:.2f} tonnes/ha</h1>
            <p>Total Production: {prediction * area:.2f} tonnes</p>
        </div>
        """, unsafe_allow_html=True)
        
        
        # ============================
        # ✅ NEW ADDITION — DOWNLOAD REPORT
        # ============================
        report = f"""
Crop Yield Prediction Report
----------------------------

Crop: {crop}
State: {state}
Season: {season}
Year: {crop_year}

Predicted Yield: {prediction:.2f} tonnes per hectare
Total Production: {prediction * area:.2f} tonnes

Inputs Used:
- Area: {area} ha
- Rainfall: {rainfall} mm
- Fertilizer: {fertilizer} kg
- Pesticide: {pesticide} kg

Model Accuracy (R²): {model_package['metrics']['r2']:.3f}
"""

        st.download_button(
            label="📥 Download Prediction Report",
            data=report,
            file_name="yield_prediction_report.txt",
            mime="text/plain"
        )
        # ============================

        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🏛️ Schemes", "💡 Tips", "📈 Comparisons"])
        
        with tab1:
            st.metric("🌾 Total Production", f"{prediction * area:.2f} tonnes")
            fig = create_feature_impact_chart(model_package)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            schemes = get_government_schemes(state, crop)
            for s in schemes:
                st.markdown(f"<div class='scheme-box'>{s}</div>", unsafe_allow_html=True)
        
        with tab3:
            tips = get_yield_improvement_tips(prediction, crop, input_data)
            for t in tips:
                st.markdown(f"<div class='tip-box'>{t}</div>", unsafe_allow_html=True)
        
        with tab4:
            fig2 = create_yield_comparison_chart(prediction, crop)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)


def model_performance_page(df):
    st.header("📊 Model Performance")
    mp = train_model(df)
    
    st.metric("R² Score", f"{mp['metrics']['r2']:.3f}")
    st.metric("MAE", f"{mp['metrics']['mae']:.3f}")
    st.metric("RMSE", f"{mp['metrics']['rmse']:.3f}")


def analytics_page(df):
    st.header("📈 Analytics Dashboard")
    
    st.metric("🌾 Total Crops", len(df["Crop"].unique()))
    st.metric("🗺️ States", len(df["State"].unique()))


if __name__ == "__main__":
    main()
