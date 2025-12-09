# 🌱 Sustainable Agriculture Yield Predictor

## Overview

This project presents an AI-powered Crop Yield Prediction application built with Streamlit, enabling farmers and agricultural stakeholders to predict crop yields, gain insights into influencing factors, and access personalized recommendations for sustainable farming practices. The application covers comprehensive features including data overview, advanced model training, interactive yield prediction, and detailed analytics.

## Features

*   **Interactive Dashboard:** User-friendly interface for seamless navigation.
*   **Data Overview:** Visualize and understand the underlying agricultural dataset.
*   **Advanced Model Training:** Train and evaluate powerful machine learning models (e.g., Random Forest, XGBoost) to predict crop yields based on various environmental and agricultural factors.
*   **Yield Prediction:** Input specific conditions (e.g., N, P, K values, pH, rainfall, temperature, area, crop, state, season) to get a precise yield prediction.
*   **Personalized Insights:**
    *   **Feature Impact Analysis:** Understand which factors most influence the predicted yield.
    *   **Relevant Government Schemes:** Access information on government initiatives applicable to the selected crop and state.
    *   **Yield Improvement Tips:** Get tailored advice for maximizing crop output and improving sustainability.
*   **Analytics:** Explore historical trends and compare predicted yields.

## Getting Started

Follow these instructions to set up and run the Sustainable Agriculture Yield Predictor application.

### Prerequisites

Make sure you have Python 3.8+ installed on your system. It's recommended to use a virtual environment.

### 1. Clone the repository

```bash
git clone https://github.com/sharvitKashikar/FarmEasy.git
cd FarmEasy
```

### 2. Set up a virtual environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate   # On Windows
```

### 3. Install Required Libraries

First, install the basic dependencies:

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

For enhanced model training capabilities, including Gradient Boosting models like XGBoost, LightGBM, and hyperparameter optimization, you can run the `setup_ultra.py` script:

```bash
python "not in use/setup_ultra.py"
```

This script will install:
*   `xgboost`
*   `lightgbm`
*   `optuna` (for hyperparameter optimization)
*   `catboost`

### 4. Prepare Your Data

The application expects a CSV file named `crop_yield.csv` in the root directory. This dataset should contain features relevant for crop yield prediction. Ensure it has columns such as `N`, `P`, `K`, `pH`, `rainfall`, `temperature`, `area_in_hectares`, `crop`, `state`, `season`, and `yield_in_tonnes`.

***Example `crop_yield.csv` structure:***

```csv
N,P,K,pH,rainfall,temperature,area_in_hectares,crop,state,season,yield_in_tonnes
90,42,43,6.5,200,20.8,5,rice,Maharashtra,Monsoon,8.5
85,58,41,6.2,210,21.5,7,rice,Maharashtra,Monsoon,8.9
... (more data)
```

### 5. Run the Application

Once all dependencies are installed and your `crop_yield.csv` is in place, run the Streamlit application:

```bash
streamlit run sustainable_agriculture_app.py
```

Your browser will automatically open to the Streamlit application, usually at `http://localhost:8501`.

## Usage

The application is divided into several sections accessible via the sidebar navigation:

### 📊 Data Overview

This section provides an interactive look at the dataset used for training, including:
*   Descriptive statistics of numerical features.
*   Distribution plots for key variables.
*   Correlation matrix to understand relationships between features.

### 🤖 Advanced Model Training

Here, you can:
*   Select features for model training.
*   Choose a regression model (e.g., Random Forest).
*   Train the model and view its performance metrics (R², MSE, MAE).
*   Optionally, enable cross-validation for more robust evaluation.

### 🔮 Yield Prediction

This is the core prediction interface. Input the following parameters:
*   **Soil Nutrients:** Nitrogen (N), Phosphorus (P), Potassium (K).
*   **Soil pH:** Acidity/Alkalinity of the soil.
*   **Environmental Factors:** Rainfall, Temperature.
*   **Land Use:** Area in Hectares.
*   **Crop Details:** Select the `Crop`, `State`, and `Season`.

After inputting, click 'Predict Yield' to get:
*   **Predicted Yield:** The estimated yield in tonnes per hectare.
*   **Feature Importance/Impact:** A breakdown of which input features contributed most to the prediction.
*   **Relevant Government Schemes:** Information on governmental support programs.
*   **Personalized Improvement Tips:** Advice tailored to increase your yield.
*   **Yield Comparison:** Compare your predicted yield with average yields.
*   **Projected Trend:** A simulated historical trend for the predicted crop.

### 📈 Analytics

Explore different analytical views and visualizations related to crop yields and influencing factors.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make changes, and submit pull requests.