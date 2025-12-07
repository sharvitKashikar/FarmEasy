# Getting Started with FarmEasy Sustainable Agriculture Yield Predictor

This guide will walk you through setting up and running the FarmEasy AI-Powered Crop Yield Predictor, including its advanced features for enhanced prediction capabilities.

## 1. Introduction

The FarmEasy application is a Streamlit-based tool designed to help farmers and agricultural stakeholders predict crop yields using AI, visualize data, train advanced models, and receive personalized recommendations. It integrates data overview, advanced machine learning model training, yield prediction, analytics, and offers 'ultra-advanced' features for higher accuracy.

## 2. Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
*   **pip**: Python package installer (usually comes with Python)

## 3. Basic Setup and Running the Application

To run the core FarmEasy application, follow these steps:

1.  **Clone the Repository (if not already done):**

    ```bash
    git clone https://github.com/sharvitKashikar/FarmEasy.git
    cd FarmEasy
    ```

2.  **Install Core Dependencies:**

    The application relies on `streamlit` and other basic data science libraries. You will need a `requirements.txt` file (not provided in this commit, but assumed for a functional app). Assuming a `requirements.txt` exists in the `FarmEasy` root directory, install them:

    ```bash
    pip install -r requirements.txt
    ```

    *Self-correction: If no `requirements.txt` is available, you might need to manually install `streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`, `shap`.* For now, we'll assume a `requirements.txt` is in place.

3.  **Place Data File:**

    Ensure your `crop_yield.csv` data file is in the root directory (or the directory from which you run the `app.py`). Without this, the application will show an error.

4.  **Run the Main Application:**

    ```bash
    streamlit run not_in_use/app.py
    ```

    This will open the application in your web browser, typically at `http://localhost:8501`.

### Application Pages (Basic Version)

Upon launching, you will see a sidebar navigation with the following pages:

*   **📊 Data Overview**: Explore your crop yield dataset, view statistics, and visualizations.
*   **🤖 Advanced Model Training**: Train machine learning models on your data.
*   **🔮 Yield Prediction**: Make predictions based on input features.
*   **📈 Analytics**: View overall analytics and insights.

## 4. Setting up Ultra-Advanced ML Features

The FarmEasy application has an 'ultra-advanced' mode that leverages powerful machine learning libraries like XGBoost, LightGBM, Optuna, and CatBoost for potentially higher prediction accuracy and more sophisticated feature analysis.

### 4.1. Install Ultra-Advanced Libraries

To enable these features, you need to run the `setup_ultra.py` script:

1.  **Navigate to the project directory:**

    ```bash
    cd FarmEasy
    ```

2.  **Run the setup script:**

    ```bash
    python not_in_use/setup_ultra.py
    ```

    This script will attempt to install the following packages:
    *   `xgboost>=1.7.0`
    *   `lightgbm>=3.3.0`
    *   `optuna>=3.0.0` (for hyperparameter optimization)
    *   `catboost>=1.1.0`

    You will see a summary of packages installed successfully.

### 4.2. Running the Ultra-Advanced Application

After installing the advanced libraries, you would typically run `app_ultra.py` to access the enhanced functionalities. 

```bash
streamlit run not_in_use/app_ultra.py
```

This `app_ultra.py` contains expanded features beyond the basic `app.py`.

### Ultra-Advanced Features Overview

The `app_ultra.py` provides an enriched 'Yield Prediction' experience, with tabs for:

*   **Feature Impact**: Visualize the contribution of each input feature to the prediction using SHAP values.
*   **Relevant Government Schemes**: Get information on government schemes pertinent to your selected state and crop.
*   **Personalized Improvement Tips**: Receive tailored advice to improve your crop yield based on predictions and input data.
*   **Yield Comparison & Historical Trend**: Compare your predicted yield against regional averages (simulated) and view projected historical trends for the crop.

## 5. Data Requirements

The application expects a `crop_yield.csv` file. Ensure this file is properly formatted and contains the necessary features for yield prediction. The specific columns expected will depend on the models used within the application. 

## 6. Troubleshooting

*   **`Failed to load data` error**: Ensure `crop_yield.csv` is present in the directory where you run `streamlit run`.
*   **Package installation issues**: Check your internet connection and ensure your Python environment is correctly set up. You might need administrative privileges to install some packages.
*   **Missing `requirements.txt`**: If the `pip install -r requirements.txt` command fails, manually install `streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`, `shap`, `xgboost`, `lightgbm`, `optuna`, `catboost` if you intend to run the ultra version.

For further assistance, please refer to the project's issue tracker or contact the developers.
