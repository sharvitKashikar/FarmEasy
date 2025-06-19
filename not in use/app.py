import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🌱 Sustainable Agriculture Yield Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .prediction-result {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .improvement-box {
        background: linear-gradient(90deg, #FF6B6B, #FF8E53);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the crop yield dataset"""
    try:
        df = pd.read_csv('crop_yield.csv')
        # Remove any rows with missing yield values
        df = df.dropna(subset=['Yield'])
        # Remove outliers using IQR method
        Q1 = df['Yield'].quantile(0.25)
        Q3 = df['Yield'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['Yield'] >= lower_bound) & (df['Yield'] <= upper_bound)]
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def engineer_features(df):
    """Enhanced feature engineering"""
    data = df.copy()
    
    # Create new features
    data['Fertilizer_per_Area'] = data['Fertilizer'] / (data['Area'] + 1e-6)
    data['Pesticide_per_Area'] = data['Pesticide'] / (data['Area'] + 1e-6)
    data['Rainfall_Fertilizer_Ratio'] = data['Annual_Rainfall'] / (data['Fertilizer'] + 1e-6)
    data['Production_per_Area'] = data['Production'] / (data['Area'] + 1e-6)
    
    # Logarithmic transformations for skewed features
    data['Log_Area'] = np.log1p(data['Area'])
    data['Log_Fertilizer'] = np.log1p(data['Fertilizer'])
    data['Log_Pesticide'] = np.log1p(data['Pesticide'])
    data['Log_Production'] = np.log1p(data['Production'])
    
    # Seasonal encoding (cyclical)
    season_mapping = {'Kharif': 1, 'Rabi': 2, 'Summer': 3, 'Whole Year': 4}
    data['Season_Numeric'] = data['Season'].map(season_mapping).fillna(0)
    data['Season_Sin'] = np.sin(2 * np.pi * data['Season_Numeric'] / 4)
    data['Season_Cos'] = np.cos(2 * np.pi * data['Season_Numeric'] / 4)
    
    # Year-based features
    data['Years_Since_Start'] = data['Crop_Year'] - data['Crop_Year'].min()
    
    return data

@st.cache_data
def preprocess_data(df):
    """Enhanced preprocessing with feature engineering"""
    # Engineer features first
    data = engineer_features(df)
    
    # Initialize label encoders
    label_encoders = {}
    
    # Encode categorical variables
    categorical_columns = ['Crop', 'Season', 'State']
    for col in categorical_columns:
        le = LabelEncoder()
        data[col + '_encoded'] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    # Enhanced feature set
    feature_columns = [
        'Crop_encoded', 'Crop_Year', 'Season_encoded', 'State_encoded', 
        'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide',
        'Fertilizer_per_Area', 'Pesticide_per_Area', 'Rainfall_Fertilizer_Ratio',
        'Log_Area', 'Log_Fertilizer', 'Log_Pesticide',
        'Season_Sin', 'Season_Cos', 'Years_Since_Start'
    ]
    
    X = data[feature_columns]
    y = data['Yield']
    
    return X, y, label_encoders, feature_columns

@st.cache_resource
def train_advanced_models(X, y):
    """Train advanced ML models with hyperparameter tuning"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = RobustScaler()  # More robust to outliers
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=12)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    selected_features = selector.get_support()
    
    # Advanced models with hyperparameter tuning
    models = {
        'Optimized Random Forest': RandomForestRegressor(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Optimized Gradient Boosting': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Ridge Regression': Ridge(alpha=1.0),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    }
    
    model_results = {}
    trained_models = {}
    
    # Train and evaluate models
    for name, model in models.items():
        if name in ['Ridge Regression', 'Elastic Net', 'SVR']:
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross-validation score
        if name in ['Ridge Regression', 'Elastic Net', 'SVR']:
            cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        model_results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        trained_models[name] = model
    
    # Create ensemble model
    ensemble_models = [
        ('rf', models['Optimized Random Forest']),
        ('gb', models['Optimized Gradient Boosting']),
        ('et', models['Extra Trees'])
    ]
    
    ensemble = VotingRegressor(estimators=ensemble_models)
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    
    ensemble_mse = mean_squared_error(y_test, y_pred_ensemble)
    ensemble_r2 = r2_score(y_test, y_pred_ensemble)
    ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
    ensemble_cv = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='r2')
    
    model_results['Ensemble Model'] = {
        'model': ensemble,
        'mse': ensemble_mse,
        'r2': ensemble_r2,
        'mae': ensemble_mae,
        'rmse': np.sqrt(ensemble_mse),
        'cv_mean': ensemble_cv.mean(),
        'cv_std': ensemble_cv.std()
    }
    
    # Select best model based on cross-validation R²
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['cv_mean'])
    best_model = model_results[best_model_name]['model']
    
    return best_model, model_results, scaler, best_model_name, selector, selected_features

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Sustainable Agriculture Yield Predictor</h1>
        <p>AI-Powered Crop Yield Prediction for Sustainable Farming</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if 'crop_yield.csv' exists in the current directory.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["📊 Data Overview", "🤖 Advanced Model Training", "🔮 Yield Prediction", "📈 Analytics"])
    
    if page == "📊 Data Overview":
        data_overview_page(df)
    elif page == "🤖 Advanced Model Training":
        advanced_model_training_page(df)
    elif page == "🔮 Yield Prediction":
        prediction_page(df)
    elif page == "📈 Analytics":
        analytics_page(df)

def data_overview_page(df):
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Records</h3>
            <h2 style="color: #2E8B57;">{:,}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Unique Crops</h3>
            <h2 style="color: #2E8B57;">{}</h2>
        </div>
        """.format(df['Crop'].nunique()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>States Covered</h3>
            <h2 style="color: #2E8B57;">{}</h2>
        </div>
        """.format(df['State'].nunique()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Years Span</h3>
            <h2 style="color: #2E8B57;">{}-{}</h2>
        </div>
        """.format(df['Crop_Year'].min(), df['Crop_Year'].max()), unsafe_allow_html=True)
    
    st.subheader("📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📊 Data Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='Yield', nbins=50, title='Distribution of Crop Yield')
        fig.update_layout(xaxis_title="Yield", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_crops = df['Crop'].value_counts().head(10)
        fig = px.bar(x=top_crops.index, y=top_crops.values, title='Top 10 Crops by Frequency')
        fig.update_layout(xaxis_title="Crop", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

def advanced_model_training_page(df):
    st.header("🤖 Advanced Model Training & Evaluation")
    
    # Show improvements
    st.markdown("""
    <div class="improvement-box">
        <h3>🚀 Enhanced Features for Better Accuracy</h3>
        <p>✅ Advanced Feature Engineering • ✅ Hyperparameter Tuning • ✅ Ensemble Methods • ✅ Feature Selection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Preprocess data
    X, y, label_encoders, feature_columns = preprocess_data(df)
    
    if st.button("🚀 Train Advanced Models", type="primary"):
        with st.spinner("Training advanced models with optimization... This may take a moment."):
            best_model, model_results, scaler, best_model_name, selector, selected_features = train_advanced_models(X, y)
            
            # Save the trained model and preprocessing objects
            joblib.dump({
                'model': best_model,
                'scaler': scaler,
                'label_encoders': label_encoders,
                'feature_columns': feature_columns,
                'best_model_name': best_model_name,
                'selector': selector,
                'selected_features': selected_features
            }, 'crop_yield_advanced_model.pkl')
            
            st.success(f"✅ Advanced models trained successfully! Best model: {best_model_name}")
            
            # Display model comparison
            st.subheader("📊 Enhanced Model Performance Comparison")
            
            results_df = pd.DataFrame(model_results).T
            results_df = results_df[['r2', 'cv_mean', 'cv_std', 'mse', 'mae', 'rmse']]
            results_df.columns = ['Test R² Score', 'CV R² Mean', 'CV R² Std', 'MSE', 'MAE', 'RMSE']
            
            st.dataframe(results_df.round(4), use_container_width=True)
            
            # Highlight best performing model
            best_r2 = results_df['Test R² Score'].max()
            best_cv = results_df['CV R² Mean'].max()
            
            st.markdown(f"""
            <div class="prediction-result">
                <h3>🎯 Best Performance Achieved!</h3>
                <p>Test R² Score: <strong>{best_r2:.4f}</strong> | CV R² Score: <strong>{best_cv:.4f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualize model performance
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(x=results_df.index, y=results_df['Test R² Score'], 
                            title='Model Test R² Score Comparison')
                fig.update_layout(xaxis_title="Model", yaxis_title="R² Score")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(x=results_df.index, y=results_df['CV R² Mean'], 
                            title='Model Cross-Validation R² Score')
                fig.update_layout(xaxis_title="Model", yaxis_title="CV R² Score")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance for tree-based models
            if best_model_name not in ['Ridge Regression', 'Elastic Net', 'SVR']:
                st.subheader("🎯 Feature Importance Analysis")
                
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', 
                               orientation='h', title='Top 10 Most Important Features')
                    st.plotly_chart(fig, use_container_width=True)

def prediction_page(df):
    st.header("🔮 Enhanced Crop Yield Prediction")
    
    # Check if advanced model exists, fallback to basic model
    model_file = 'crop_yield_advanced_model.pkl'
    try:
        model_data = joblib.load(model_file)
        st.success(f"✅ Advanced model loaded! Using: {model_data['best_model_name']}")
        is_advanced = True
    except FileNotFoundError:
        try:
            model_data = joblib.load('crop_yield_model.pkl')
            st.info(f"ℹ️ Using basic model: {model_data['best_model_name']}. Train advanced models for better accuracy!")
            is_advanced = False
        except FileNotFoundError:
            st.error("❌ No trained model found. Please train a model first.")
            return
    
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    best_model_name = model_data['best_model_name']
    
    st.subheader("🌾 Enter Crop Details for Prediction")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        crop = st.selectbox("Select Crop", df['Crop'].unique())
        season = st.selectbox("Select Season", df['Season'].unique())
        state = st.selectbox("Select State", df['State'].unique())
        crop_year = st.number_input("Crop Year", min_value=1997, max_value=2030, value=2024)
    
    with col2:
        area = st.number_input("Area (hectares)", min_value=0.0, value=1000.0, step=100.0)
        annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1200.0, step=50.0)
        fertilizer = st.number_input("Fertilizer Used (kg)", min_value=0.0, value=50000.0, step=1000.0)
        pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0, value=100.0, step=10.0)
    
    if st.button("🎯 Predict Yield", type="primary"):
        try:
            if is_advanced:
                # Create enhanced input data for advanced model
                input_dict = {
                    'Crop_encoded': label_encoders['Crop'].transform([crop])[0],
                    'Crop_Year': crop_year,
                    'Season_encoded': label_encoders['Season'].transform([season])[0],
                    'State_encoded': label_encoders['State'].transform([state])[0],
                    'Area': area,
                    'Annual_Rainfall': annual_rainfall,
                    'Fertilizer': fertilizer,
                    'Pesticide': pesticide,
                    'Fertilizer_per_Area': fertilizer / (area + 1e-6),
                    'Pesticide_per_Area': pesticide / (area + 1e-6),
                    'Rainfall_Fertilizer_Ratio': annual_rainfall / (fertilizer + 1e-6),
                    'Log_Area': np.log1p(area),
                    'Log_Fertilizer': np.log1p(fertilizer),
                    'Log_Pesticide': np.log1p(pesticide),
                    'Season_Sin': np.sin(2 * np.pi * label_encoders['Season'].transform([season])[0] / 4),
                    'Season_Cos': np.cos(2 * np.pi * label_encoders['Season'].transform([season])[0] / 4),
                    'Years_Since_Start': crop_year - 1997
                }
                
                input_data = pd.DataFrame([input_dict])
                
                # Use selector if available
                if 'selector' in model_data and best_model_name in ['Ridge Regression', 'Elastic Net', 'SVR']:
                    input_scaled = scaler.transform(input_data)
                    input_selected = model_data['selector'].transform(input_scaled)
                    prediction = model.predict(input_selected)[0]
                elif best_model_name in ['Ridge Regression', 'Elastic Net', 'SVR']:
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)[0]
                else:
                    prediction = model.predict(input_data)[0]
            else:
                # Basic model prediction
                input_data = pd.DataFrame({
                    'Crop_encoded': [label_encoders['Crop'].transform([crop])[0]],
                    'Crop_Year': [crop_year],
                    'Season_encoded': [label_encoders['Season'].transform([season])[0]],
                    'State_encoded': [label_encoders['State'].transform([state])[0]],
                    'Area': [area],
                    'Annual_Rainfall': [annual_rainfall],
                    'Fertilizer': [fertilizer],
                    'Pesticide': [pesticide]
                })
                
                if best_model_name == 'Linear Regression':
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)[0]
                else:
                    prediction = model.predict(input_data)[0]
            
            # Ensure non-negative prediction
            prediction = max(0, prediction)
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-result">
                <h2>🎯 Predicted Yield</h2>
                <h1>{prediction:.3f} tonnes per hectare</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate total production
            total_production = prediction * area
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📦 Total Production", f"{total_production:,.2f} tonnes")
            with col2:
                st.metric("🌾 Yield per Hectare", f"{prediction:.3f} tonnes/ha")
            with col3:
                efficiency = (prediction / (fertilizer/area + 1e-6)) * 1000 if area > 0 else 0
                st.metric("⚡ Efficiency", f"{efficiency:.2f}")
            
            # Show input summary
            st.subheader("📋 Input Summary")
            summary_data = {
                'Parameter': ['Crop', 'Season', 'State', 'Year', 'Area (ha)', 
                             'Rainfall (mm)', 'Fertilizer (kg)', 'Pesticide (kg)'],
                'Value': [crop, season, state, crop_year, f"{area:,}", 
                         f"{annual_rainfall:,}", f"{fertilizer:,}", f"{pesticide:,}"]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

def analytics_page(df):
    st.header("📈 Advanced Analytics")
    
    # Yield trends over time
    st.subheader("📊 Yield Trends Over Time")
    yearly_yield = df.groupby('Crop_Year')['Yield'].mean().reset_index()
    fig = px.line(yearly_yield, x='Crop_Year', y='Yield', 
                  title='Average Crop Yield Over Years')
    fig.update_layout(xaxis_title="Year", yaxis_title="Average Yield")
    st.plotly_chart(fig, use_container_width=True)
    
    # State-wise analysis
    st.subheader("🗺️ State-wise Yield Analysis")
    state_yield = df.groupby('State')['Yield'].mean().sort_values(ascending=False).head(15)
    fig = px.bar(x=state_yield.values, y=state_yield.index, orientation='h',
                title='Top 15 States by Average Yield')
    fig.update_layout(xaxis_title="Average Yield", yaxis_title="State")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    numeric_cols = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
    correlation_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                   title='Correlation Matrix of Numeric Variables')
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal analysis
    st.subheader("🌅 Seasonal Yield Patterns")
    seasonal_yield = df.groupby('Season')['Yield'].mean().reset_index()
    fig = px.pie(seasonal_yield, values='Yield', names='Season',
                title='Average Yield Distribution by Season')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 