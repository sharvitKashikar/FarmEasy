# Advanced Sustainable Agriculture Yield Predictor

The `not in use/app.py` and `not in use/app_ultra.py` scripts represent more advanced and elaborate versions of a crop yield prediction system, compared to `sustainable_agriculture_app.py`. They incorporate a wider range of machine learning models and data preprocessing techniques for potentially higher accuracy and more detailed analysis.

## `app.py` - Comprehensive Yield Predictor

### Features
-   Supports multiple regression models: RandomForest, GradientBoosting, ExtraTrees, VotingRegressor, Linear Regression, Ridge, Lasso, ElasticNet, DecisionTree, SVR.
-   Advanced preprocessing with `LabelEncoder`, `StandardScaler`, `RobustScaler`, `PolynomialFeatures`.
-   Feature selection using `SelectKBest` and `RFE`.
-   Hyperparameter tuning with `GridSearchCV`.
-   Detailed performance metrics (MSE, R2, MAE).
-   Extensive data visualizations using Plotly.

### How to Use
1.  **Prerequisites**: Install numerous ML libraries. Use `pip install streamlit pandas numpy scikit-learn plotly xgboost lightgbm catboost optuna`. Although `setup_ultra.py` covers some, it's advisable to ensure all in `app.py` are present.
2.  A `crop_yield.csv` dataset is required.
3.  Run the application:
    ```bash
    streamlit run "not in use/app.py"
    ```
4.  Explore the various prediction modes, feature engineering, and model training options via the Streamlit interface.

## `app_ultra.py` - Ultra-Advanced Yield Predictor

### Features
-   Similar to `app.py` but potentially optimized or extended with more sophisticated logic or models.
-   Emphasizes custom CSS for an enhanced user experience.
-   Focuses on robust model selection and evaluation.

### How to Use
1.  **Prerequisites**: Installation of advanced ML libraries is crucial. Refer to `setup_ultra.py` documentation.
2.  A `crop_yield.csv` dataset is required.
3.  Run the application:
    ```bash
    streamlit run "not in use/app_ultra.py"
    ```

## `setup_ultra.py` - Setup Script for Advanced ML Libraries

This script is provided to help install the specific, more advanced machine learning libraries required by applications like `app_ultra.py`. It automates the installation of packages such as `xgboost`, `lightgbm`, `optuna`, and `catboost`.

### How to Use
1.  Run the script directly from your terminal:
    ```bash
    python "not in use/setup_ultra.py"
    ```
2.  The script will attempt to install each listed package. It will report success or failure for each.

### Recommended Usage
It is recommended to run `setup_ultra.py` before attempting to use advanced yield predictor applications like `app_ultra.py` to ensure all necessary dependencies are met for optimal performance and functionality.

## Note on 'not in use' Directory
The presence of these files in a `not in use` directory suggests they might be experimental, alternative, or deprecated versions. Users should understand that `sustainable_agriculture_app.py` might be the primary intended application, while `app.py` and `app_ultra.py` provide more advanced, possibly less maintained, or alternative approaches.