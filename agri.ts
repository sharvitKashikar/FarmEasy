// agriYield.ts

// Interface for soil and weather data
interface CropInput {
  soilMoisture: number; // percentage (0–100)
  soilPH: number;       // pH value (0–14)
  rainfall: number;    // mm per season
  temperature: number; // °C
}

// Function to estimate crop yield
function estimateCropYield(input: CropInput): number {
  let yieldScore = 0;

  // Soil Moisture
  if (input.soilMoisture >= 40 && input.soilMoisture <= 70) {
    yieldScore += 30;
  } else {
    yieldScore += 10;
  }

  // Soil pH
  if (input.soilPH >= 6 && input.soilPH <= 7.5) {
    yieldScore += 25;
  } else {
    yieldScore += 10;
  }

  // Rainfall
  if (input.rainfall >= 600 && input.rainfall <= 1200) {
    yieldScore += 25;
  } else {
    yieldScore += 10;
  }

  // Temperature
  if (input.temperature >= 20 && input.temperature <= 30) {
    yieldScore += 20;
  } else {
    yieldScore += 5;
  }

  // Convert score into yield (kg/hectare)
  const estimatedYield = yieldScore * 50;
  return estimatedYield;
}

// Sample input
const wheatData: CropInput = {
  soilMoisture: 55,
  soilPH: 6.8,
  rainfall: 800,
  temperature: 26,
};

// Output
const yieldResult = estimateCropYield(wheatData);
console.log(`🌾 Estimated Crop Yield: ${yieldResult} kg/hectare`);
