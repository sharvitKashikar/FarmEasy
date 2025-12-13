# 🌾 Soil Nutrient Deficiency Detector

This system, implemented in `worldagri.py`, is a Streamlit application designed to detect soil nutrient deficiencies (Nitrogen, Phosphorus, Potassium) and provide fertilizer recommendations.

## Features
- Predicts N, P, K levels (Low/Medium/High) based on pH, Organic Carbon, and Moisture.
- Provides specific fertilizer recommendations and doses.
- Uses pre-trained RandomForestClassifier models for each nutrient.

## How to Run
1.  **Dependencies**: Ensure you have `streamlit`, `pandas`, `numpy`, `scikit-learn`, and `plotly` installed:
    ```bash
    pip install streamlit pandas numpy scikit-learn plotly
    ```
2.  **Dataset**: Make sure `soil_data.csv` is available in the same directory as `worldagri.py`. This CSV must contain columns for N_level, P_level, K_level, pH, Organic_Carbon, and Moisture.
3.  **Execute**: Run the Streamlit application from your terminal:
    ```bash
    streamlit run worldagri.py
    ```
    This will open the application in your web browser.

## Usage
- **Input Soil Values**: Enter values for pH (0-14), Organic Carbon (%), and Moisture (%).
- **Detect Deficiency**: Click the 'Detect Deficiencies & Recommend Fertilizer' button.
- **Results**: The application will display the predicted nutrient levels for N, P, and K, along with tailored fertilizer recommendations and dosage.