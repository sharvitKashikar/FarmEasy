# Smart Irrigation Advisor
# Agriculture-based Python Project

def base_water_requirement(crop):
    crop = crop.lower()
    water_map = {
        "rice": 1200,
        "wheat": 600,
        "maize": 500,
        "cotton": 700,
        "sugarcane": 1500,
        "potato": 550
    }
    return water_map.get(crop, 600)


def calculate_water_need(crop, temperature, rainfall):
    base_water = base_water_requirement(crop)

    # Temperature factor
    if temperature > 35:
        temp_factor = 1.2
    elif temperature < 20:
        temp_factor = 0.9
    else:
        temp_factor = 1.0

    # Rainfall adjustment
    effective_rainfall = rainfall * 0.8

    total_water = (base_water * temp_factor) - effective_rainfall
    return max(total_water, 0)


def irrigation_advice(water_needed):
    if water_needed > 800:
        return "Irrigate frequently (every 2–3 days)"
    elif 400 <= water_needed <= 800:
        return "Moderate irrigation (every 4–6 days)"
    else:
        return "Low irrigation (every 7–10 days)"


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
