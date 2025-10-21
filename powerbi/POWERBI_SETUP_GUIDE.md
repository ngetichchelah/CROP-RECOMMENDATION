# Power BI Dashboard Setup Guide

## Step 1: Import Data

1. Open **Power BI Desktop**
2. Click **Get Data** → **Text/CSV**
3. Import these files (one by one):
   - `data/processed/crops_with_predictions.csv`
   - `data/processed/crop_requirements.csv`
   - `data/processed/model_performance.csv`
   - `data/processed/crop_categories.csv`

## Step 2: Create Relationships

Go to **Model View** (left sidebar):

1. Drag `crops_with_predictions[label]` → `crop_requirements[crop]`
2. Drag `crops_with_predictions[label]` → `crop_categories[Crop]`

## Step 3: Create DAX Measures

Go to **Data View**, click **New Measure**, and add these:

Total_Samples = COUNTROWS(crops_with_predictions)

Total_Crops = DISTINCTCOUNT(crops_with_predictions[label])

Avg_Accuracy = AVERAGE(model_performance[Accuracy])

Correct_Predictions = SUM(crops_with_predictions[correct_prediction])

Prediction_Accuracy = 
DIVIDE([Correct_Predictions], [Total_Samples], 0)

Avg_Confidence = AVERAGE(crops_with_predictions[confidence])

Avg_Nitrogen = AVERAGE(crops_with_predictions[N])

Avg_Phosphorus = AVERAGE(crops_with_predictions[P])

Avg_Potassium = AVERAGE(crops_with_predictions[K])

NPK_Category = 
VAR N_Level = IF([Avg_Nitrogen] < 40, "Low", IF([Avg_Nitrogen] < 80, "Medium", "High"))
VAR P_Level = IF([Avg_Phosphorus] < 40, "Low", IF([Avg_Phosphorus] < 80, "Medium", "High"))
VAR K_Level = IF([Avg_Potassium] < 40, "Low", IF([Avg_Potassium] < 80, "Medium", "High"))
RETURN N_Level & " N, " & P_Level & " P, " & K_Level & " K"

Climate_Type = 
SWITCH(
    TRUE(),
    AVERAGE(crops_with_predictions[temperature]) < 20 && AVERAGE(crops_with_predictions[rainfall]) < 100, "Cool & Dry",
    AVERAGE(crops_with_predictions[temperature]) < 20 && AVERAGE(crops_with_predictions[rainfall]) >= 100, "Cool & Wet",
    AVERAGE(crops_with_predictions[temperature]) >= 20 && AVERAGE(crops_with_predictions[rainfall]) < 100, "Warm & Dry",
    "Warm & Wet"
)
Most_Recommended_Crop = 
TOPN(1, VALUES(crops_with_predictions[predicted_crop]), COUNTROWS(crops_with_predictions), DESC)
