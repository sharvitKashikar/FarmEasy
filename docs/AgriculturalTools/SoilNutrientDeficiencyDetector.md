# Soil Nutrient Deficiency Detector

The `worldagri.py` script provides a Streamlit web application that detects deficiencies in Nitrogen (N), Phosphorus (P), and Potassium (K) based on soil pH, organic carbon, and moisture. It then generates tailored fertilizer recommendations.

## Features
-   Predicts N, P, K levels (Low, Medium, High) using `RandomForestClassifier`.
-   Provides fertilizer recommendations including approximate dosage.
-   Visualizes input data and predictions.
-   User-friendly interface using Streamlit.

## How to Use
1.  Ensure required libraries are installed (`pip install streamlit pandas numpy scikit-learn plotly`).
2.  A `soil_data.csv` dataset is needed, which must contain columns for `N_level`, `P_level`, `K_level` (categorical: Low/Medium/High), `pH`, `Organic_Carbon`, and `Moisture`.
3.  Run the application from your terminal:
    ```bash
    streamlit run worldagri.py
    ```
4.  Open the application in your browser.
5.  Input the soil parameters in the sidebar.
6.  Click 'Detect Nutrient Levels & Recommend' to get the analysis.

## Inputs
-   **Soil pH**
-   **Organic Carbon** (percentage)
-   **Moisture** (percentage)

## Example Prediction
Assume input: pH=6.5, Organic Carbon=1.5, Moisture=30.

Expected output will be an analysis of N, P, K levels (e.g., N: Medium, P: Low, K: High) and corresponding fertilizer recommendations (e.g., 'Apply moderate fertilizer dose for N', 'Apply high-dose fertilizer for P').

## Code Snippet (Core Logic)
```python
# ... (imports and dataset loading)

@st.cache_resource
def train_models(df):
    features = ["pH", "Organic_Carbon", "Moisture"]
    models = {}
    scalers = {}

    for nutrient in ["N", "P", "K"]:
        X = df[features]
        y = df[nutrient + "_level"] # categorical: Low/Medium/High

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=250, random_state=42)
        model.fit(X_train, y_train)

        models[nutrient] = model
        scalers[nutrient] = scaler

    return models, scalers

def main():
    # ... (Streamlit UI for inputs)
    if st.sidebar.button("🔍 Detect Nutrient Levels & Recommend"):
        # ... (input processing and prediction)
        st.subheader("📊 Predicted Nutrient Levels")
        # ... (display levels and recommendations)
```