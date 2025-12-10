# AI Crop Recommendation System

This document describes the AI-powered Crop Recommendation System, a Streamlit application designed to assist farmers and agricultural stakeholders in making informed decisions about crop cultivation. By leveraging machine learning, the system predicts crop yields based on environmental and soil conditions, and offers personalized recommendations, relevant government schemes, and yield improvement tips.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.x installed. The application relies on several Python libraries which can be installed using pip.

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

### Running the Application

1.  **Download the application file:** Obtain `sustainable_agriculture_app.py`.
2.  **Navigate to the directory:** Open your terminal or command prompt and change the directory to where `sustainable_agriculture_app.py` is located.
3.  **Run Streamlit:** Execute the following command:

    ```bash
    streamlit run sustainable_agriculture_app.py
    ```

    This will open the application in your web browser.

## ✨ Features

The Sustainable Agriculture Yield Predictor offers the following key features:

### 1. Interactive Yield Prediction
Users can input various parameters to get a predicted crop yield. The UI is designed as follows:

*   **Input Fields:**
    *   **Soil Nutrient Ratios:** Nitrogen (N), Phosphorus (P), Potassium (K) (slider input: 0-140).
    *   **Environmental Factors:** Temperature (slider input: 0-40), Humidity (slider input: 0-100), pH (slider input: 0-14), Rainfall (slider input: 0-300).
    *   **Location & Crop:** State (dropdown selection), Crop (dropdown selection).

*   **Prediction Button:** Triggers the yield prediction based on the entered inputs.

### 2. Prediction Insights
Upon prediction, the application displays:

*   **Predicted Yield:** The estimated yield in tonnes/ha.
*   **Feature Importance Chart:** A bar chart illustrating which input features had the most significant impact on the prediction (e.g., rainfall, temperature, NPK ratios).

### 3. Relevant Government Schemes
Based on the selected state and recommended crop, the system provides information on applicable government agricultural schemes. (This feature is currently simulated with placeholder data.)

### 4. Personalized Improvement Tips
The application offers customized tips to improve yield, tailored to the predicted yield and input conditions. (This feature is currently simulated with placeholder data.)

### 5. Yield Comparison and Historical Trends
*   **Comparison Chart:** Visualizes the predicted yield against average or historical yields for the selected crop and state.
*   **Projected Trend:** A simulated line chart showing the projected yield trend over several years.

## ⚙️ How it Works (Technical Overview)

### Data Loading and Preprocessing
*   The application loads data from `crop_yield.csv`.
*   Numerical features are scaled using `StandardScaler`.
*   Categorical features like `District_Name` (derived from `State` and `Crop`) are encoded using `LabelEncoder`.

### Model Training
*   A `RandomForestRegressor` model is trained on the historical crop yield data.
*   The model learns the relationships between environmental factors, soil nutrients, and crop yields.

### Prediction
*   When a user provides inputs, these are preprocessed and fed to the trained `RandomForestRegressor` model to generate a yield prediction.

### External Dependencies (Optional)

The repository contains other files like `not in use/app_ultra.py` and `not in use/setup_ultra.py`. `setup_ultra.py` suggests installing advanced ML libraries (XGBoost, LightGBM, Optuna, CatBoost) for potentially more sophisticated models or analyses. While the main `sustainable_agriculture_app.py` uses `RandomForestRegressor`, these files indicate possibilities for future enhancements or alternative model explorations.

## ⚠️ Troubleshooting

*   **`FileNotFoundError` for `crop_yield.csv`:** Ensure `crop_yield.csv` is present in the same directory as `sustainable_agriculture_app.py` or provide the correct path.
*   **Package Installation Issues:** Verify your internet connection and Python environment if `pip install` commands fail.