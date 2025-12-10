# Fertilizer Recommendation System
# Agriculture-based Python Test Code

def nutrient_status(value, nutrient):
    if nutrient == "N":  # Nitrogen
        if value < 40:
            return "Low"
        elif 40 <= value <= 80:
            return "Optimal"
        else:
            return "High"

    elif nutrient == "P":  # Phosphorus
        if value < 20:
            return "Low"
        elif 20 <= value <= 50:
            return "Optimal"
        else:
            return "High"

    elif nutrient == "K":  # Potassium
        if value < 50:
            return "Low"
        elif 50 <= value <= 120:
            return "Optimal"
        else:
            return "High"


def fertilizer_advice(n_status, p_status, k_status):
    advice = []

    if n_status == "Low":
        advice.append("Add Urea or Ammonium Sulphate")
    if p_status == "Low":
        advice.append("Add Single Super Phosphate (SSP)")
    if k_status == "Low":
        advice.append("Add Muriate of Potash (MOP)")

    if not advice:
        return "Soil nutrients are balanced. No fertilizer required."

    return ", ".join(advice)


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
