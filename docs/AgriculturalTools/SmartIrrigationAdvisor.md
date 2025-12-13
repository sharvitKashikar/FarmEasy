# Smart Irrigation Advisor

The `test3.py` script is a command-line Python program that advises on irrigation needs for various crops, taking into account temperature and rainfall.

## Features
-   Calculates estimated water requirement for specific crops.
-   Adjusts water needs based on temperature and effective rainfall.
-   Provides frequency-based irrigation advice.

## How to Use
1.  Run the script from your terminal:
    ```bash
    python test3.py
    ```
2.  The program will prompt you to enter the crop name, average temperature, and rainfall.
3.  It will then output the estimated water requirement and irrigation advice.

## Inputs
-   **Crop Name**: (e.g., `rice`, `wheat`, `maize`, `cotton`, `sugarcane`, `potato`)
-   **Average Temperature** (°C)
-   **Rainfall** (mm)

## Example Interaction
```
🌾 Smart Crop Irrigation Advisor 🌾
Enter crop name: Wheat
Enter average temperature (°C): 25
Enter rainfall (mm): 100

📊 Irrigation Report
Crop: Wheat
Estimated Water Requirement: 520.00 mm
Irrigation Advice: Moderate irrigation (every 4–6 days)
```

## Code Snippet (Core Logic)
```python
# ... (base_water_requirement, calculate_water_need, irrigation_advice functions)

# -------- Main Program --------
print("🌾 Smart Crop Irrigation Advisor 🌾")

crop = input("Enter crop name: ")
temperature = float(input("Enter average temperature (°C): "))
rainfall = float(input("Enter rainfall (mm): "))

water_needed = calculate_water_need(crop, temperature, rainfall)
advice = irrigation_advice(water_needed)

print("\n📊 Irrigation Report")
print("Crop:", crop.capitalize())
print(f"Estimated Water Requirement: {water_needed:.2f} mm")
print("Irrigation Advice:", advice)
```