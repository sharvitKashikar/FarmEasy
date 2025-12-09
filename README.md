# FarmEasy

FarmEasy is a collection of applications designed to assist farmers with various tasks, from pest detection to soil nutrient analysis.

## Project Structure

```
FarmEasy/
├── .venv/
├── README.md
├── requirements.txt
├── main.py
├── apps/
│   ├── pest_detection_app.py
│   ├── soil_nutrient_deficiency_detector.py
│   └── ...
└── models/
    └── ...
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sharvitKashikar/FarmEasy.git
    cd FarmEasy
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Available Applications

FarmEasy includes several Streamlit applications. To run an application, navigate to the project root and use the `streamlit run` command.

### 1. Pest Detection App

This application allows users to upload images of crops and detect potential pests using a pre-trained model.

**How to run:**

```bash
streamlit run apps/pest_detection_app.py
```

### 2. Soil Nutrient Deficiency Detector

This application helps farmers analyze their soil's nutrient levels and provides fertilizer recommendations. Users can input soil test values (e.g., pH, Nitrogen, Phosphorus, Potassium), and the app will predict nutrient levels and suggest appropriate fertilizers based on a trained Random Forest model.

**How to run:**

```bash
streamlit run apps/soil_nutrient_deficiency_detector.py
```

## Contributing

We welcome contributions! Please feel free to fork the repository, make changes, and submit pull requests.

## License

This project is licensed under the MIT License.