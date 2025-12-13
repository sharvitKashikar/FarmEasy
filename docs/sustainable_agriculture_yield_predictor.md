# 🌱 Sustainable Agriculture Yield Predictor

This application, implemented in `sustainable_agriculture_app.py`, is a Streamlit-based tool for predicting crop yield and offering sustainable agriculture insights.

## Features
- Predicts crop yield based on various environmental and agricultural inputs.
- Provides visualizations for data analysis.
- Offers tips for sustainable practices and government schemes.
- Uses a RandomForestRegressor model for predictions.

## How to Run
1.  **Dependencies**: Ensure you have `streamlit`, `pandas`, `numpy`, `plotly`, and `scikit-learn` installed. You can install them using pip:
    ```bash
    pip install streamlit pandas numpy plotly scikit-learn
    ```
2.  **Dataset**: Make sure `crop_yield.csv` is available in the same directory as `sustainable_agriculture_app.py`.
3.  **Execute**: Run the Streamlit application from your terminal:
    ```bash
    streamlit run sustainable_agriculture_app.py
    ```
    This will open the application in your web browser.

## Usage
- **Data Upload**: Upload your `crop_yield.csv` file (if not already present).
- **Input Parameters**: Use the sidebar to input various parameters like crop type, season, soil type, N, P, K values, pH, rainfall, temperature, area, and pesticides.
- **Yield Prediction**: The application will display the predicted yield and provide insights.
- **Sustainable Tips & Schemes**: Explore sections for best practices and government schemes related to sustainable agriculture.

### Advanced Versions (not in use folder)
Note that `not in use/app.py` and `not in use/app_ultra.py` contain more advanced versions of this predictor with additional features, model ensembles, and hyperparameter tuning capabilities. Documentation for these could be created separately if they are to be brought into active use.