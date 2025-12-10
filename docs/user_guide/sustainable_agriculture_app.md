# Sustainable Agriculture Yield Predictor: User Guide

This document provides a comprehensive guide to using the AI-powered Sustainable Agriculture Yield Predictor Streamlit application. This tool assists farmers and agricultural enthusiasts in making informed decisions by predicting crop yields and offering recommendations based on various environmental and soil conditions.

## 1. Introduction

The Sustainable Agriculture Yield Predictor is an AI-driven web application designed to help optimize crop yields. By inputting specific environmental and soil parameters, users can receive predictions for different crops grown in various regions, along with personalized improvement tips and relevant government schemes.

### Key Features:
*   **Interactive Input:** Easily enter soil nutrients (Nitrogen, Phosphorus, Potassium), temperature, humidity, pH, rainfall, state, crop type, and season.
*   **AI-Powered Predictions:** Get accurate crop yield predictions based on a trained machine learning model.
*   **Feature Importance:** Understand which input factors most significantly influence the yield prediction.
*   **Personalized Tips:** Receive actionable advice for improving yield based on predictions.
*   **Government Schemes:** Discover relevant government agricultural schemes for your region and crop.
*   **Yield Comparison & Trends:** Visualize predicted yield against regional averages and historical trends.

## 2. Getting Started

To run the application, you need to have Python and `pip` installed. 

### Prerequisites
Make sure you have the following Python packages installed:

```bash
pip install streamlit pandas numpy plotly scikit-learn
# Additional packages for advanced models (if available and used, e.g., xgboost, lightgbm)
# pip install xgboost lightgbm
```

### Running the Application

1.  **Save the application:** Ensure the `sustainable_agriculture_app.py` file is saved to your local machine.
2.  **Open your terminal or command prompt:** Navigate to the directory where you saved the file.
3.  **Execute the Streamlit command:**
    ```bash
    streamlit run sustainable_agriculture_app.py
    ```

    This will open the application in your default web browser.

## 3. Application Workflow

Upon launching the application, you will see a main header and a sidebar for navigation. The primary functionality resides in the 'Yield Prediction' section.

### Header

The application starts with an engaging header:

```html
<div class="main-header">
    <h1>🌱 Sustainable Agriculture Yield Predictor</h1>
    <p>AI-Powered Crop Yield Prediction for Sustainable Farming</p>
</div>
```

### Inputting Conditions for Prediction

To get a crop yield prediction, navigate to the 'Yield Prediction' section (if not already there). You will find various input fields:

*   **Soil Nutrient Levels:**
    *   `Nitrogen (N)`: Enter the nitrogen content in your soil.
    *   `Phosphorus (P)`: Enter the phosphorus content.
    *   `Potassium (K)`: Enter the potassium content.
*   **Environmental Conditions:**
    *   `Temperature`: Average temperature in Celsius.
    *   `Humidity`: Average relative humidity.
    *   `pH Value`: Soil pH level.
    *   `Rainfall`: Total rainfall in mm.
*   **Location & Crop Details:**
    *   `State`: Select the agricultural state.
    *   `Crop`: Select the specific crop you are interested in.
    *   `Season`: Select the growing season.

After entering all the required information, click the 'Predict Yield' button to generate predictions.

### Understanding the Prediction Results

Once you click 'Predict Yield', the application will display the results in several tabs, providing a holistic view of the prediction and related information.

#### A. Yield Prediction & Feature Impact

This tab will show your predicted crop yield and a chart illustrating the impact of each input feature on that prediction. This helps you understand which factors are most critical for the current prediction.

#### B. Relevant Government Schemes

This section lists government schemes pertinent to the selected state and crop. These schemes can provide financial assistance, subsidies, or other support for farmers.

```html
<div class="scheme-box">
    {scheme_details_here}
</div>
```

#### C. Personalized Improvement Tips

Based on the predicted yield and input conditions, this section offers actionable tips to help improve your crop's yield. These tips are tailored to the specific context.

```html
<div class="tip-box">
    {tip_details_here}
</div>
```

#### D. Yield Comparison & Trend

This tab provides visual insights:

*   **Yield Comparison Chart:** Compares your predicted yield with regional averages or benchmarks.
*   **Projected Yield Trend:** Shows a simulated historical and projected future yield trend for the selected crop.

## 4. Model Training & Analytics (Advanced Users)

The application may include sections for 'Advanced Model Training' and 'Analytics' (as suggested by `app.py`). These sections are typically for developers or data scientists to:

*   **Train custom models:** Use the dataset to train or re-train the underlying machine learning models.
*   **Perform deeper data analysis:** Explore data distributions, correlations, and model performance metrics.

Refer to the specific UI within the application for detailed instructions on using these advanced features.

## 5. Troubleshooting

*   **Data Loading Errors:** If you see an error like "Failed to load data. Please check if 'crop_yield.csv' exists...":
    *   Ensure that the `crop_yield.csv` file is present in the same directory as `sustainable_agriculture_app.py`.
    *    Verify the file is not corrupted and is accessible.
*   **Missing Libraries:** If you encounter `ModuleNotFoundError`:
    *   Ensure all required libraries (`streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`) are installed using `pip`.

For any other issues, re-check your environment setup and the application logs in your terminal.