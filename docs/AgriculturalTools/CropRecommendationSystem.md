# Crop Recommendation System

The `agri2.py` script implements an AI-based Crop Recommendation System using Streamlit. This tool helps farmers determine the most suitable crop to grow by analyzing various soil and climate parameters.

## Features
- Predicts optimal crop based on Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, and Humidity.
- Utilizes a `RandomForestClassifier` trained on a `crop_recommendation.csv` dataset.
- User-friendly web interface via Streamlit.

## How to Use
1.  Ensure you have Streamlit and scikit-learn installed (`pip install streamlit pandas numpy scikit-learn`).
2.  Place the `crop_recommendation.csv` dataset in the same directory as `agri2.py`.
3.  Run the application from your terminal:
    ```bash
    streamlit run agri2.py
    ```
4.  Access the application in your web browser (usually `http://localhost:8501`).
5.  Enter the required soil and climate values in the sidebar.
6.  Click 'Recommend Crop' to get the prediction.

## Inputs
The system requires the following inputs, typically entered via the Streamlit sidebar:
-   **Nitrogen (N)**: Amount of Nitrogen in the soil (kg/ha)
-   **Phosphorus (P)**: Amount of Phosphorus in the soil (kg/ha)
-   **Potassium (K)**: Amount of Potassium in the soil (kg/ha)
-   **Soil pH**: pH value of the soil
-   **Rainfall**: Average rainfall (mm)
-   **Temperature**: Average temperature (°C)
-   **Humidity**: Average humidity (%)

## Example Prediction
Assume the following inputs:
-   N: 50, P: 40, K: 40, pH: 6.5, Rainfall: 120, Temp: 25, Humidity: 60

The system will output a recommended crop based on its trained model.

## Code Snippet (Core Logic)
```python
# ... (imports and setup)

@st.cache_resource
def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, scaler, acc

def main():
    # ... (Streamlit UI setup)
    inputs = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    scaled_inputs = scaler.transform(inputs)
    prediction = model.predict(scaled_inputs)
    st.success(f"The best crop to grow is: **{prediction[0]}**")
    # ... (rest of main)
```