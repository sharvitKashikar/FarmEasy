# Fertilizer Recommendation System

The `nutri.py` script is a basic command-line Python program that recommends fertilizers based on the Nitrogen (N), Phosphorus (P), and Potassium (K) levels in the soil.

## Features
-   Analyzes N, P, K levels and categorizes them as Low, Optimal, or High.
-   Provides specific fertilizer advice (e.g., Urea for low N, SSP for low P, MOP for low K).

## How to Use
1.  Run the script from your terminal:
    ```bash
    python nutri.py
    ```
2.  The program will prompt you to enter values for Nitrogen, Phosphorus, and Potassium.
3.  After inputting the values, it will display a soil nutrient analysis and a fertilizer recommendation.

## Inputs
-   **Nitrogen (N)**: Value in kg/ha (e.g., `60`)
-   **Phosphorus (P)**: Value in kg/ha (e.g., `30`)
-   **Potassium (K)**: Value in kg/ha (e.g., `80`)

## Example Interaction
```
🌿 Fertilizer Recommendation System 🌿
Enter Nitrogen value (kg/ha): 30
Enter Phosphorus value (kg/ha): 15
Enter Potassium value (kg/ha): 40

📊 Soil Nutrient Analysis
Nitrogen: Low
Phosphorus: Low
Potassium: Low
Fertilizer Recommendation: Add Urea or Ammonium Sulphate, Add Single Super Phosphate (SSP), Add Muriate of Potash (MOP)
```

## Code Snippet (Core Logic)
```python
# ... (nutrient_status and fertilizer_advice functions)

# -------- Main Program --------
print("🌿 Fertilizer Recommendation System 🌿")

N = float(input("Enter Nitrogen value (kg/ha): "))
P = float(input("Enter Phosphorus value (kg/ha): "))
K = float(input("Enter Potassium value (kg/ha): "))

n_status = nutrient_status(N, "N")
p_status = nutrient_status(P, "P")
k_status = nutrient_status(K, "K")

recommendation = fertilizer_advice(n_status, p_status, k_status)

print("\n📊 Soil Nutrient Analysis")
print("Nitrogen:", n_status)
print("Phosphorus:", p_status)
print("Potassium:", k_status)
print("Fertilizer Recommendation:", recommendation)
```