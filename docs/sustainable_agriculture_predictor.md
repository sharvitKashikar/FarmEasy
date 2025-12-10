# 🌱 Sustainable Agriculture Yield Predictor

This document provides an overview and setup guide for the AI-Powered Crop Yield Prediction application, designed to support sustainable farming practices. The application offers functionalities ranging from data overview and advanced model training to yield prediction and analytics, incorporating relevant government schemes and personalized improvement tips.

## 🚀 Getting Started

To run the Sustainable Agriculture Yield Predictor, ensure you have Python installed, along with the required libraries. This application is built using Streamlit.

### Prerequisites

*   Python 3.7+
*   Access to a `crop_yield.csv` file, which should contain your agricultural data for analysis and prediction.

### Installation

1.  **Clone the Repository (if not already done):**

    ```bash
    git clone https://github.com/sharvitKashikar/FarmEasy.git
    cd FarmEasy
    ```

2.  **Install Basic Dependencies:**

    First, ensure you have Streamlit and core ML libraries installed:

    ```bash
    pip install streamlit pandas numpy plotly scikit-learn
    ```

3.  **Install Ultra-Advanced ML Libraries (Optional but Recommended):**

    The application can leverage advanced machine learning models for higher accuracy. To install these, run the `setup_ultra.py` script:

    ```bash
    python "not in use/setup_ultra.py"
    ```

    This script will install packages like `xgboost`, `lightgbm`, `optuna`, and `catboost`.

### Running the Application

After ensuring all dependencies are met and `crop_yield.csv` is available in your working directory, you can launch the Streamlit application:

```bash
streamlit run sustainable_agriculture_app.py
```

Your application will open in a web browser.

## 📊 Application Features

The 'Sustainable Agriculture Yield Predictor' offers several pages accessible via the sidebar navigation:

### 1. Data Overview

Provides an initial look at your `crop_yield.csv` dataset, including basic statistics and visualizations.

### 2. Advanced Model Training

Allows users to train and evaluate advanced machine learning models for yield prediction. This section leverages the 'ultra-advanced ML libraries' if installed, offering choices like XGBoost, LightGBM, and CatBoost, along with hyperparameter tuning provided by Optuna.

### 3. Yield Prediction

Enables users to input various environmental and agricultural parameters to get a predicted crop yield. This page also presents:

*   **Feature Impact Chart:** Visualizes which input features contributed most to the prediction.
*   **Relevant Government Schemes:** Provides information on government initiatives applicable to the predicted crop and state (based on simulated data).
*   **Personalized Improvement Tips:** Offers tailored advice to enhance crop yield given the prediction and input data.
*   **Yield Comparison & Historical Trend:** Compares the predicted yield with averages and shows a projected historical trend.

### 4. Analytics

Further analytical tools and visualizations to understand model performance and data patterns.

## ⭐ Customization and Styling

The application uses custom CSS for an enhanced user interface, defining styles for headers, prediction boxes, metric cards, and information boxes for tips and government schemes. These styles are defined within `sustainable_agriculture_app.py`.