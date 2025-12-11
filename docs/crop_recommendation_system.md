# 🌾 Crop Recommendation System

This system, implemented in `agri2.py`, is a Streamlit application designed to recommend the best crop to grow given specific soil and climate conditions.

## Features
- Predicts crop based on Nitrogen (N), Phosphorus (P), Potassium (K) levels, pH, rainfall, temperature, and humidity.
- Uses a pre-trained RandomForestClassifier model.
- User-friendly interface via Streamlit.

## How to Run
1.  **Dependencies**: Ensure you have `streamlit`, `pandas`, `numpy`, and `scikit-learn` installed. You can install them using pip:
    ```bash
    pip install streamlit pandas numpy scikit-learn
    ```
2.  **Dataset**: Make sure `crop_recommendation.csv` is available in the same directory as `agri2.py`.
3.  **Execute**: Run the Streamlit application from your terminal:
    ```bash
    streamlit run agri2.py
    ```
    This will open the application in your web browser.

## Usage
On the left sidebar, enter the following soil and climate values:
- **Nitrogen (N)** (0-200)
- **Phosphorus (P)** (0-200)
- **Potassium (K)** (0-200)
- **Soil pH** (3.0-10.0)
- **Rainfall (mm)** (0-500)
- **Temperature (°C)** (10-50)
- **Humidity (%)** (10-100)

Click the '🌾 Recommend Crop' button to get the recommended crop.