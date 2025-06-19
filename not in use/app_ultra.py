import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
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
    .warning-box {
        background: linear-gradient(135deg, #ff9a9e, #fecfef);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #721c24;
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
        df['Rainfall_Category'] = pd.cut(df['Annual_Rainfall'], 
                                       bins=[0, 500, 1000, 1500, float('inf')], 
                                       labels=['Low', 'Medium', 'High', 'Very High'])
        
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
    
    # Add encoded features
    features.extend(['Crop_encoded', 'Season_encoded', 'State_encoded', 'Crop_Year'])
    
    X = df[features]
    y = df['Yield']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest (best balance of accuracy and interpretability)
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
    
    # Create model package
    model_package = {
        'model': rf_model,
        'scaler': scaler,
        'label_encoders': {
            'Crop': le_crop,
            'Season': le_season,
            'State': le_state
        },
        'features': features,
        'metrics': {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        },
        'test_data': {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    }
    
    return model_package

def get_government_schemes(state, crop):
    """Get relevant government schemes based on state and crop"""
    schemes = {
        'National': [
            "🌾 **Pradhan Mantri Fasal Bima Yojana (PMFBY)** - Crop insurance scheme",
            "💰 **PM-KISAN** - ₹6,000 annual income support to farmer families",
            "🚜 **Sub-Mission on Agricultural Mechanization** - Subsidies on farm equipment",
            "🌱 **National Mission for Sustainable Agriculture** - Climate-resilient farming",
            "💧 **Pradhan Mantri Krishi Sinchayee Yojana** - Irrigation efficiency improvement"
        ]
    }
    
    # State-specific schemes
    state_schemes = {
        'Punjab': [
            "🌾 **Punjab Crop Diversification Program** - Incentives to shift from rice-wheat",
            "💧 **Free Electricity for Agriculture** - Subsidized power for farming"
        ],
        'Uttar Pradesh': [
            "🌱 **UP Crop Diversification Scheme** - Support for alternative crops",
            "🚜 **Kisan Credit Card** - Easy agricultural loans"
        ],
        'Maharashtra': [
            "☔ **Maharashtra Drought Relief Package** - Support during water scarcity",
            "🍯 **Honey Mission** - Beekeeping promotion for additional income"
        ],
        'West Bengal': [
            "🌾 **Krishak Bandhu** - Financial assistance to farmers",
            "🐟 **Blue Revolution** - Fisheries development support"
        ]
    }
    
    # Crop-specific schemes
    crop_schemes = {
        'Rice': [
            "🌾 **Rice Fortification Program** - Quality improvement initiatives",
            "🔬 **System of Rice Intensification (SRI)** - Water-saving rice cultivation"
        ],
        'Wheat': [
            "🌾 **National Food Security Mission** - Wheat productivity enhancement",
            "🏪 **Minimum Support Price (MSP)** - Guaranteed procurement price"
        ],
        'Cotton': [
            "🌱 **Cotton Technology Mission** - Improved cotton varieties",
            "🐛 **Integrated Pest Management** - Sustainable pest control"
        ]
    }
    
    all_schemes = schemes['National'].copy()
    if state in state_schemes:
        all_schemes.extend(state_schemes[state])
    if crop in crop_schemes:
        all_schemes.extend(crop_schemes[crop])
    
    return all_schemes

def get_yield_improvement_tips(prediction, crop, inputs):
    """Generate personalized tips to improve yield"""
    tips = []
    
    # General tips based on prediction level
    if prediction < 2:
        tips.append("🚨 **Low Yield Alert**: Consider soil testing and nutrient management")
        tips.append("💡 **Immediate Action**: Check soil pH and organic matter content")
    elif prediction < 5:
        tips.append("📈 **Moderate Yield**: Good potential for improvement with optimization")
    else:
        tips.append("🌟 **Excellent Yield**: Maintain current practices and consider value addition")
    
    # Fertilizer recommendations
    fert_per_area = inputs['Fertilizer_per_Area']
    if fert_per_area < 20:
        tips.append("🌱 **Increase Fertilizer**: Consider 20-25% increase in balanced NPK fertilizers")
        tips.append("🧪 **Soil Testing**: Get soil tested for precise nutrient requirements")
    elif fert_per_area > 100:
        tips.append("⚠️ **Reduce Fertilizer**: Over-fertilization detected. Reduce by 15-20%")
        tips.append("🌿 **Organic Matter**: Add compost to improve soil health")
    
    # Water management
    if inputs['Annual_Rainfall'] < 600:
        tips.append("💧 **Water Management**: Install drip irrigation for water efficiency")
        tips.append("🌧️ **Rainwater Harvesting**: Collect and store rainwater for dry periods")
    elif inputs['Annual_Rainfall'] > 2000:
        tips.append("🌊 **Drainage**: Ensure proper drainage to prevent waterlogging")
    
    # Crop-specific tips
    crop_tips = {
        'Rice': [
            "🌾 **SRI Method**: Try System of Rice Intensification for 20-30% yield increase",
            "💧 **Alternate Wetting**: Practice alternate wetting and drying to save water"
        ],
        'Wheat': [
            "🌾 **Seed Rate**: Use 100-125 kg/ha seed rate for optimal plant population",
            "❄️ **Winter Care**: Protect from frost during grain filling stage"
        ],
        'Cotton': [
            "🌱 **Bt Cotton**: Consider Bt varieties for better pest resistance",
            "🐛 **IPM**: Implement Integrated Pest Management practices"
        ],
        'Sugarcane': [
            "🌿 **Ratoon Management**: Proper ratoon care can increase successive crop yields",
            "🚜 **Mechanization**: Use machinery for timely operations"
        ]
    }
    
    if crop in crop_tips:
        tips.extend(crop_tips[crop])
    
    # Sustainability tips
    tips.extend([
        "🌱 **Cover Crops**: Plant legumes during off-season to improve soil nitrogen",
        "🦋 **Biodiversity**: Maintain field borders with native plants for beneficial insects",
        "📊 **Record Keeping**: Maintain detailed farm records for better decision making",
        "🤝 **Farmer Groups**: Join local FPOs for better input procurement and marketing"
    ])
    
    return tips

def create_yield_comparison_chart(prediction, crop, state):
    """Create comparison chart with regional averages"""
    
    # Sample regional data (in practice, this would come from your dataset)
    regional_data = {
        'Rice': {'Punjab': 6.2, 'West Bengal': 5.8, 'Andhra Pradesh': 5.5, 'National': 4.2},
        'Wheat': {'Punjab': 5.1, 'Uttar Pradesh': 3.2, 'Haryana': 4.8, 'National': 3.5},
        'Cotton': {'Gujarat': 2.1, 'Maharashtra': 1.8, 'Telangana': 2.3, 'National': 1.9},
        'Sugarcane': {'Maharashtra': 78, 'Uttar Pradesh': 65, 'Karnataka': 82, 'National': 70}
    }
    
    if crop in regional_data:
        data = regional_data[crop]
        regions = list(data.keys())
        yields = list(data.values())
        
        # Add user's prediction
        regions.append('Your Prediction')
        yields.append(prediction)
        
        colors = ['lightblue'] * (len(regions)-1) + ['red']
        
        fig = px.bar(x=regions, y=yields, title=f'{crop} Yield Comparison (tonnes/ha)',
                    color=regions, color_discrete_sequence=colors)
        fig.update_layout(showlegend=False, height=400)
        return fig
    
    return None

def create_feature_impact_chart(model_package, inputs):
    """Create chart showing impact of different factors on yield"""
    
    model = model_package['model']
    features = model_package['features']
    
    # Get feature importances
    importances = model.feature_importances_
    feature_names = [f.replace('_', ' ').title() for f in features]
    
    fig = px.bar(x=importances, y=feature_names, orientation='h',
                title='Factors Affecting Crop Yield (Feature Importance)',
                labels={'x': 'Importance Score', 'y': 'Factors'})
    fig.update_layout(height=500)
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Sustainable Agriculture Yield Predictor</h1>
        <p>AI-Powered Crop Yield Prediction with Government Schemes & Improvement Tips</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🔮 Yield Prediction", 
        "📊 Model Performance", 
        "📈 Analytics Dashboard",
        "🌱 Sustainability Tips"
    ])
    
    if page == "🔮 Yield Prediction":
        prediction_page(df)
    elif page == "📊 Model Performance":
        model_performance_page(df)
    elif page == "📈 Analytics Dashboard":
        analytics_page(df)
    elif page == "🌱 Sustainability Tips":
        sustainability_page(df)

def prediction_page(df):
    st.header("🔮 Crop Yield Prediction")
    
    # Train model
    with st.spinner("Training AI model..."):
        model_package = train_model(df)
    
    st.success(f"✅ Model trained! Accuracy: {model_package['metrics']['r2']:.3f} (R² Score)")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        crop = st.selectbox("Select Crop", sorted(df['Crop'].unique()))
        season = st.selectbox("Select Season", sorted(df['Season'].unique()))
        state = st.selectbox("Select State", sorted(df['State'].unique()))
        crop_year = st.number_input("Crop Year", min_value=1997, max_value=2030, value=2024)
    
    with col2:
        area = st.number_input("Area (hectares)", min_value=1.0, value=100.0, step=10.0)
        annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=800.0, step=50.0)
        fertilizer = st.number_input("Fertilizer Used (kg)", min_value=0.0, value=5000.0, step=100.0)
        pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0, value=50.0, step=5.0)
    
    if st.button("🎯 Predict Yield", type="primary"):
        
        # Prepare input data
        input_data = {
            'Area': area,
            'Annual_Rainfall': annual_rainfall,
            'Fertilizer': fertilizer,
            'Pesticide': pesticide,
            'Fertilizer_per_Area': fertilizer / area,
            'Pesticide_per_Area': pesticide / area,
            'Crop_encoded': model_package['label_encoders']['Crop'].transform([crop])[0],
            'Season_encoded': model_package['label_encoders']['Season'].transform([season])[0],
            'State_encoded': model_package['label_encoders']['State'].transform([state])[0],
            'Crop_Year': crop_year
        }
        
        # Create DataFrame and scale
        input_df = pd.DataFrame([input_data])
        input_scaled = model_package['scaler'].transform(input_df[model_package['features']])
        
        # Make prediction
        prediction = model_package['model'].predict(input_scaled)[0]
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Predicted Yield</h2>
            <h1>{prediction:.2f} tonnes per hectare</h1>
            <p>Total Production: {prediction * area:.2f} tonnes</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different information
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🏛️ Government Schemes", "💡 Improvement Tips", "📈 Comparisons"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Metrics
                total_production = prediction * area
                efficiency = (prediction / (fertilizer/area + 1e-6)) * 1000 if area > 0 else 0
                sustainability_score = min(100, max(0, 100 - abs(fertilizer/area - 50)))
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Production Metrics</h3>
                    <p><strong>Total Production:</strong> {total_production:.2f} tonnes</p>
                    <p><strong>Efficiency Score:</strong> {efficiency:.2f}</p>
                    <p><strong>Sustainability Score:</strong> {sustainability_score:.1f}/100</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Feature impact chart
                fig_impact = create_feature_impact_chart(model_package, input_data)
                st.plotly_chart(fig_impact, use_container_width=True)
        
        with tab2:
            # Government schemes
            schemes = get_government_schemes(state, crop)
            st.markdown("### 🏛️ Relevant Government Schemes")
            
            for scheme in schemes:
                st.markdown(f"""
                <div class="scheme-box">
                    {scheme}
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            # Improvement tips
            tips = get_yield_improvement_tips(prediction, crop, input_data)
            st.markdown("### 💡 Personalized Improvement Tips")
            
            for tip in tips:
                st.markdown(f"""
                <div class="tip-box">
                    {tip}
                </div>
                """, unsafe_allow_html=True)
        
        with tab4:
            # Yield comparison
            fig_comparison = create_yield_comparison_chart(prediction, crop, state)
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Historical trend (simulated)
            years = list(range(2019, 2025))
            trend_yields = [prediction * (0.95 + 0.02 * i) for i in range(len(years))]
            
            fig_trend = px.line(x=years, y=trend_yields, 
                              title=f'Projected {crop} Yield Trend',
                              labels={'x': 'Year', 'y': 'Yield (tonnes/ha)'})
            st.plotly_chart(fig_trend, use_container_width=True)

def model_performance_page(df):
    st.header("📊 Model Performance Analysis")
    
    # Train model
    model_package = train_model(df)
    metrics = model_package['metrics']
    test_data = model_package['test_data']
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 R² Score", f"{metrics['r2']:.3f}")
    with col2:
        st.metric("📊 MAE", f"{metrics['mae']:.3f}")
    with col3:
        st.metric("📈 RMSE", f"{metrics['rmse']:.3f}")
    with col4:
        st.metric("🔄 CV Score", f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    
    # Prediction vs Actual plot
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter = px.scatter(
            x=test_data['y_test'], 
            y=test_data['y_pred'],
            title='Actual vs Predicted Yield',
            labels={'x': 'Actual Yield (tonnes/ha)', 'y': 'Predicted Yield (tonnes/ha)'}
        )
        
        # Add perfect prediction line
        min_val = min(test_data['y_test'].min(), test_data['y_pred'].min())
        max_val = max(test_data['y_test'].max(), test_data['y_pred'].max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Residual plot
        residuals = test_data['y_test'] - test_data['y_pred']
        fig_residual = px.scatter(
            x=test_data['y_pred'], y=residuals,
            title='Residual Plot',
            labels={'x': 'Predicted Yield', 'y': 'Residuals'}
        )
        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_residual, use_container_width=True)

def analytics_page(df):
    st.header("📈 Agricultural Analytics Dashboard")
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌾 Total Crops", len(df['Crop'].unique()))
    with col2:
        st.metric("🗺️ States Covered", len(df['State'].unique()))
    with col3:
        st.metric("📊 Average Yield", f"{df['Yield'].mean():.2f} t/ha")
    with col4:
        st.metric("📈 Data Points", len(df))
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Yield by crop
        crop_yield = df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(10)
        fig_crop = px.bar(x=crop_yield.index, y=crop_yield.values,
                         title='Average Yield by Crop (Top 10)')
        st.plotly_chart(fig_crop, use_container_width=True)
    
    with col2:
        # Yield by state
        state_yield = df.groupby('State')['Yield'].mean().sort_values(ascending=False).head(10)
        fig_state = px.bar(x=state_yield.index, y=state_yield.values,
                          title='Average Yield by State (Top 10)')
        fig_state.update_xaxis(tickangle=45)
        st.plotly_chart(fig_state, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Factor Correlation Analysis")
    numeric_cols = ['Yield', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Matrix of Agricultural Factors")
    st.plotly_chart(fig_corr, use_container_width=True)

def sustainability_page(df):
    st.header("🌱 Sustainability & Best Practices")
    
    st.markdown("""
    <div class="tip-box">
        <h3>🌍 Sustainable Agriculture Principles</h3>
        <p>Building resilient farming systems for future generations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sustainability tabs
    tab1, tab2, tab3 = st.tabs(["🌱 Practices", "📊 Metrics", "🎯 Goals"])
    
    with tab1:
        practices = [
            "🌿 **Crop Rotation**: Alternating crops to improve soil health",
            "💧 **Water Conservation**: Drip irrigation and rainwater harvesting",
            "🐛 **Integrated Pest Management**: Biological and organic pest control",
            "🌾 **Cover Cropping**: Planting cover crops to prevent soil erosion",
            "♻️ **Composting**: Converting organic waste into natural fertilizer",
            "🌳 **Agroforestry**: Integrating trees into farming systems",
            "📊 **Precision Agriculture**: Using data for optimal resource use",
            "🦋 **Biodiversity**: Maintaining diverse ecosystems on farms"
        ]
        
        for practice in practices:
            st.markdown(f"""
            <div class="tip-box">
                {practice}
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        # Sustainability metrics
        efficiency_data = df.groupby('State')['Efficiency'].mean().sort_values(ascending=False).head(10)
        fig_efficiency = px.bar(x=efficiency_data.index, y=efficiency_data.values,
                               title='Agricultural Efficiency by State')
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    with tab3:
        st.markdown("""
        <div class="scheme-box">
            <h3>🎯 Sustainable Development Goals</h3>
            <ul>
                <li>🍽️ <strong>Zero Hunger</strong>: Ensure food security for all</li>
                <li>💧 <strong>Clean Water</strong>: Efficient water management</li>
                <li>🌱 <strong>Climate Action</strong>: Reduce agricultural emissions</li>
                <li>🌍 <strong>Life on Land</strong>: Protect terrestrial ecosystems</li>
                <li>🤝 <strong>Partnerships</strong>: Collaborate for sustainable agriculture</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 