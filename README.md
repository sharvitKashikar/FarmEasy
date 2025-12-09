# 🌱 Sustainable Agriculture Yield Predictor

**AI-Powered Crop Yield Prediction for Sustainable Farming**

This Streamlit application provides a comprehensive platform for predicting crop yields, analyzing agricultural data, and training advanced machine learning models. It aims to empower farmers and agricultural stakeholders with data-driven insights to optimize crop production and promote sustainable practices.

## ✨ Features

*   **Interactive Data Overview:** Explore your agricultural datasets with various visualizations.
*   **Advanced Model Training:** Train and evaluate sophisticated machine learning models for yield prediction, including support for advanced libraries like XGBoost and LightGBM.
*   **Yield Prediction:** Get accurate yield predictions based on input parameters and receive personalized improvement tips.
*   **Government Schemes & Analytics:** Discover relevant government schemes and explore yield comparisons and historical trends.
*   **User-Friendly Interface:** Built with Streamlit for an intuitive and responsive user experience.

## 🚀 Getting Started

Follow these steps to set up and run the `Sustainable Agriculture Yield Predictor` application locally.

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/sharvitKashikar/FarmEasy.git
cd FarmEasy
```

### 2. Install Dependencies

First, install the basic dependencies:

```bash
pip install -r requirements.txt # (Assuming a requirements.txt exists or will be created)
```

For enhanced model training capabilities, including XGBoost, LightGBM, and Optuna, we recommend installing the 'ultra-advanced' libraries. Navigate to the `not in use` directory and run the setup script:

```bash
cd "not in use"
python setup_ultra.py
cd ..
```

This script will install additional libraries that unlock advanced features and potentially higher prediction accuracy.

### 3. Prepare Data

Ensure you have your agricultural data in a CSV file named `crop_yield.csv` in the root directory where the `sustainable_agriculture_app.py` script is located. The application expects specific columns for effective analysis and prediction.

### 4. Run the Application

Once dependencies are installed and data is ready, run the Streamlit application:

```bash
streamlit run sustainable_agriculture_app.py
```

This will open the application in your default web browser.

## 💡 Usage

The application is divided into several sections, accessible via the sidebar navigation:

*   **📊 Data Overview:** View raw data, summary statistics, and initial data visualizations.
*   **🤖 Advanced Model Training:** Configure and train various machine learning models. Evaluate their performance and tune hyperparameters.
*   **🔮 Yield Prediction:** Input specific agricultural parameters (e.g., chosen crop, state, soil type, temperature, rainfall) to receive a predicted yield and associated insights like feature impact, government schemes, and personalized improvement tips.
*   **📈 Analytics:** Compare predicted yields with historical data and explore trends.

## 🛠️ Technologies Used

*   **Python**
*   **Streamlit** - For interactive web application development
*   **Pandas & NumPy** - For data manipulation
*   **Plotly & Matplotlib** - For data visualization
*   **Scikit-learn** - For machine learning models (e.g., RandomForestRegressor)
*   **XGBoost, LightGBM, Optuna, CatBoost** - For advanced model training and hyperparameter optimization (installed via `setup_ultra.py`)

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## 📄 License

This project is open-source and available under the MIT License. (Presumption, adjust if a different license is intended.)
