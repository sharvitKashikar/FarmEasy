```markdown
# 🌱 Sustainable Agriculture Yield Predictor

This repository hosts an AI-powered crop yield prediction application designed for sustainable farming practices. Built with Streamlit, it offers a user-friendly interface to analyze agricultural data, train machine learning models, predict crop yields, and gain insights through interactive visualizations.

## ✨ Features

-   **📊 Data Overview**: Explore the loaded agricultural dataset with summary statistics and visualizations.
-   **🤖 Advanced Model Training**: Train and evaluate machine learning models (e.g., RandomForestRegressor) on your data.
-   **🔮 Yield Prediction**: Input specific farm parameters to get an estimated crop yield.
-   **📈 Analytics**: Dive deeper into model performance, feature importance, and historical trends.
-   **🏛️ Government Schemes & Improvement Tips**: Get personalized suggestions and relevant government initiatives based on predictions.
-   **Custom Styling**: Enhanced user experience with a modern, gradient-based UI.

## 🚀 Getting Started

Follow these steps to set up and run the Sustainable Agriculture Yield Predictor application locally.

### Prerequisites

Make sure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Installation

Install the required Python packages:

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

### Data Preparation

The application expects a CSV file named `crop_yield.csv` in the same directory as `sustainable_agriculture_app.py`. This file should contain your agricultural data, including features like `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, `state`, `crop`, and `yield`.

An example structure for `crop_yield.csv` might look like this:

| N   | P   | K   | temperature | humidity | ph    | rainfall | state       | crop      | yield |
| :-- | :-- | :-- | :---------- | :------- | :---- | :------- | :---------- | :-------- | :---- |
| 90  | 42  | 43  | 20.87       | 82.00    | 6.50  | 202.93   | Karnataka   | rice      | 5.5   |
| 85  | 58  | 41  | 21.77       | 80.31    | 7.03  | 226.66   | Maharashtra | maize     | 4.2   |
| ... | ... | ... | ...         | ...      | ...   | ...      | ...         | ...       | ...   |

### Running the Application

Navigate to the directory containing `sustainable_agriculture_app.py` in your terminal and run:

```bash
streamlit run sustainable_agriculture_app.py
```

This will open the application in your default web browser.

## ⚙️ How it Works

-   **Data Loading & Caching**: Data is loaded from `crop_yield.csv` and cached for performance (`@st.cache_data`).
-   **Preprocessing**: Categorical features (`state`, `crop`) are encoded using `LabelEncoder`, and numerical features are scaled using `StandardScaler`.
-   **Model**: A `RandomForestRegressor` is used for yield prediction, trained on user-selected features.
-   **User Interface**: Streamlit provides an interactive UI with sliders, select boxes, and custom CSS for a visually appealing experience.

## 📁 Project Structure

-   `sustainable_agriculture_app.py`: The main Streamlit application script.
-   `crop_yield.csv` (expected): The dataset used for training and prediction.
-   `not in use/`: This directory contains older or experimental scripts that are not part of the main application workflow. They are provided for historical context or potential future development but are not actively used by `sustainable_agriculture_app.py`.

## 🤝 Contributing

Contributions are welcome! Please feel free to fork the repository, make changes, and submit pull requests.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE-optional). (Consider adding a LICENSE file)
```