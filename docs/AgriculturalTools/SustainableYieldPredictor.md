# Sustainable Agriculture Yield Predictor

The `sustainable_agriculture_app.py` script presents a Streamlit web application designed to predict crop yield and provide insights for improving agricultural practices. It's built around machine learning models to analyze various environmental and agricultural factors.

## Features
-   Predicts crop yield based on parameters like Nitrogen, Phosphorus, Potassium, Temperature, Rainfall, pH, and Area.
-   Offers best practices and government schemes relevant to sustainable agriculture.
-   Includes visualizations for data distributions and predictions.
-   Utilizes a `RandomForestRegressor` for yield prediction.

## How to Use
1.  Ensure you have the necessary libraries installed (`pip install streamlit pandas numpy scikit-learn plotly`).
2.  A `crop_yield.csv` dataset is required in the same directory as the script. This dataset should contain columns like `Crop`, `Yield`, `Nitrogen`, `Phosphorus`, `Potassium`, `Temperature`, `Rainfall`, `pH`, `Area`.
3.  Run the application from your terminal:
    ```bash
    streamlit run sustainable_agriculture_app.py
    ```
4.  Open the displayed URL in your web browser.
5.  Input the required values in the sidebar.
6.  Click 'Predict Yield' to see the predicted yield and recommendations.

## Inputs
The system takes the following user inputs:
-   **Crop Type** (dropdown selection)
-   **Nitrogen (N)** (kg/ha)
-   **Phosphorus (P)** (kg/ha)
-   **Potassium (K)** (kg/ha)
-   **Temperature** (°C)
-   **Rainfall** (mm)
-   **pH**
-   **Area** (acres/hectares, depending on dataset units)

## Example Flow
Users select a crop, input soil and climate conditions, and plantation area. The system then processes these inputs to provide a predicted yield, along with tips and relevant government schemes for agricultural improvement.

## Code Snippet (Core Logic)
```python
# ... (imports and setup)

@st.cache_data
def load_and_prepare_data():
    # ... (data loading and preprocessing including LabelEncoder and StandardScaler)
    return df, X, y, label_encoder, scaler

@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def main():
    # ... (Streamlit UI layout and input handling)
    if st.sidebar.button("🌱 Predict Yield"):
        # ... (input collection and scaling)
        prediction = model.predict(input_df_scaled)[0]
        # ... (display prediction and recommendations)
```

## Related Yield Predictor Applications
This repository also contains `not in use/app.py` and `not in use/app_ultra.py`, which are other versions of a yield prediction system with potentially more advanced models and features. The `sustainable_agriculture_app.py` likely represents a refined or simpler version for specific use cases.