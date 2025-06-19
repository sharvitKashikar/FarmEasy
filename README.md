# 🌱 Sustainable Agriculture Yield Predictor

An AI-powered web application built with Streamlit for predicting crop yields using machine learning techniques. This application helps farmers and agricultural professionals make data-driven decisions for sustainable farming practices.

## 🚀 Features

- **📊 Data Overview**: Comprehensive analysis of the crop yield dataset
- **🤖 Model Training**: Train and compare multiple ML models (Random Forest, Gradient Boosting, Linear Regression)
- **🔮 Yield Prediction**: Real-time crop yield prediction based on user inputs
- **📈 Advanced Analytics**: Interactive visualizations and insights
- **🎯 Feature Importance**: Understanding which factors most influence crop yield
- **📱 Responsive Design**: Modern, user-friendly interface

## 📋 Dataset Features

The application uses a comprehensive crop yield dataset with the following features:
- **Crop**: Type of crop (Rice, Wheat, Maize, etc.)
- **Crop_Year**: Year of cultivation
- **Season**: Growing season (Kharif, Rabi, Summer, Whole Year)
- **State**: Indian state where crop is grown
- **Area**: Area under cultivation (hectares)
- **Production**: Total production
- **Annual_Rainfall**: Annual rainfall (mm)
- **Fertilizer**: Fertilizer usage (kg)
- **Pesticide**: Pesticide usage (kg)
- **Yield**: Crop yield (target variable)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
```bash
# If using git
git clone <your-repository-url>
cd edunet-project

# Or download and extract the project files
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Dataset
Ensure `crop_yield.csv` is in the project directory.

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🎯 How to Use

### 1. Data Overview
- Navigate to "📊 Data Overview" to explore the dataset
- View statistics, sample data, and visualizations
- Understand the distribution of crops, states, and yield patterns

### 2. Train Models
- Go to "🤖 Model Training" page
- Click "🚀 Train Models" to train multiple ML models
- Compare model performance metrics (R² Score, MSE, MAE, RMSE)
- View feature importance for the best performing model

### 3. Make Predictions
- Navigate to "🔮 Yield Prediction"
- Select crop parameters:
  - Crop type
  - Season
  - State
  - Year
  - Area (hectares)
  - Annual rainfall (mm)
  - Fertilizer usage (kg)
  - Pesticide usage (kg)
- Click "🎯 Predict Yield" to get the prediction
- View predicted yield and estimated total production

### 4. Analytics
- Explore "📈 Analytics" for advanced insights
- View yield trends over time
- State-wise performance analysis
- Correlation analysis between variables
- Seasonal yield patterns

## 🧠 Machine Learning Models

The application implements and compares three ML models:

1. **Random Forest Regressor**: Ensemble method using multiple decision trees
2. **Gradient Boosting Regressor**: Sequential ensemble learning
3. **Linear Regression**: Linear relationship modeling

The best performing model (based on R² score) is automatically selected for predictions.

## 📊 Key Metrics

- **R² Score**: Proportion of variance explained by the model
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

## 🎨 Application Structure

```
edunet-project/
├── app.py                 # Main Streamlit application
├── crop_yield.csv         # Dataset
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── crop_yield_model.pkl   # Trained model (generated after training)
```

## 🔧 Customization

### Adding New Features
- Modify the `feature_columns` list in the `preprocess_data()` function
- Update the input form in `prediction_page()` accordingly

### Adding New Models
- Add new models to the `models` dictionary in `train_models()` function
- Ensure proper preprocessing for each model type

### Styling
- Modify the CSS in the `st.markdown()` section for custom styling
- Update colors, fonts, and layout as needed

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from GitHub

### Docker (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📈 Performance Tips

- **Data Caching**: The app uses `@st.cache_data` for efficient data loading
- **Model Caching**: Trained models are cached using `@st.cache_resource`
- **Large Datasets**: For very large datasets, consider data sampling or pagination

## 🐛 Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Data Loading Issues**
   - Ensure `crop_yield.csv` is in the correct directory
   - Check file permissions and encoding

3. **Model Training Errors**
   - Verify data preprocessing steps
   - Check for missing values or invalid data types

4. **Prediction Errors**
   - Ensure model is trained before making predictions
   - Verify input values are within valid ranges

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📧 Support

For issues or questions:
- Check the troubleshooting section
- Review the code comments
- Create an issue in the repository

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ for sustainable agriculture and data-driven farming decisions.** 