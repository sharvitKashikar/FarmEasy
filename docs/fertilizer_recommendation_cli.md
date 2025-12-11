# 🌿 Fertilizer Recommendation System (CLI)

This script, `nutri.py`, is a simple command-line interface (CLI) tool that provides fertilizer recommendations based on observed Nitrogen (N), Phosphorus (P), and Potassium (K) values in soil.

## How It Works
1.  **Nutrient Status**: Determines if N, P, or K levels are 'Low', 'Optimal', or 'High' based on predefined thresholds.
2.  **Fertilizer Advice**: Recommends specific fertilizers (Urea, SSP, MOP) if nutrient levels are low.

## How to Run
1.  **Dependencies**: No special dependencies beyond standard Python libraries.
2.  **Execute**: Run the script directly from your terminal:
    ```bash
    python nutri.py
    ```

## Usage
The script will prompt you to enter values for Nitrogen, Phosphorus, and Potassium (kg/ha):

```bash
🌿 Fertilizer Recommendation System 🌿
Enter Nitrogen value (kg/ha): 50
Enter Phosphorus value (kg/ha): 15
Enter Potassium value (kg/ha): 80

📊 Soil Nutrient Analysis
Nitrogen: Optimal
Phosphorus: Low
Potassium: Optimal
Fertilizer Recommendation: Add Single Super Phosphate (SSP)
```