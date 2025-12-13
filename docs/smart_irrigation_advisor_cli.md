# 💧 Smart Irrigation Advisor (CLI)

This script, `test3.py`, is a command-line interface (CLI) tool that advises on irrigation frequency based on crop type, temperature, and rainfall.

## How It Works
1.  **Base Water Requirement**: Uses predefined base water requirements for different crops.
2.  **Adjustments**: Modifies the requirement based on temperature (hotter, more water; cooler, less water) and effective rainfall (reducing the need for irrigation).
3.  **Irrigation Advice**: Provides advice on irrigation frequency (frequently, moderate, low) based on the calculated water need.

## Supported Crops
- Rice
- Wheat
- Maize
- Cotton
- Sugarcane
- Potato

## How to Run
1.  **Dependencies**: No special dependencies beyond standard Python libraries.
2.  **Execute**: Run the script directly from your terminal:
    ```bash
    python test3.py
    ```

## Usage
1.  The script will prompt you to enter the crop name, average temperature (`°C`), and rainfall (`mm`).
2.  The output will include the estimated water requirement and specific irrigation advice.

```bash
🌾 Smart Crop Irrigation Advisor 🌾
Enter crop name: wheat
Enter average temperature (°C): 30
Enter rainfall (mm): 50

📊 Irrigation Report
Crop: Wheat
Estimated Water Requirement: 560.00 mm
Irrigation Advice: Moderate irrigation (every 4–6 days)
```