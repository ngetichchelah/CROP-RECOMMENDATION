
-- CROP RECOMMENDATION SYSTEM - ANALYTICAL SQL QUERIES

-- Query 1: Basic Statistics
-- Get overview of all crops in database
SELECT 
    COUNT(*) as total_samples,
    COUNT(DISTINCT label) as total_crops
FROM crops;


-- NUTRIENT ANALYSIS QUERIES

-- Query 2: Crops for Low Nitrogen Soil (N < 50)
SELECT 
    crop,
    ROUND(N_avg, 2) as avg_nitrogen_required,
    ROUND(N_min, 2) as min_nitrogen,
    ROUND(N_max, 2) as max_nitrogen
FROM crop_requirements
WHERE N_avg < 50
ORDER BY N_avg ASC;

-- Query 3: High Nitrogen Requiring Crops (N > 80)
SELECT 
    crop,
    ROUND(N_avg, 2) as avg_nitrogen,
    ROUND(P_avg, 2) as avg_phosphorus,
    ROUND(K_avg, 2) as avg_potassium
FROM crop_requirements
WHERE N_avg > 80
ORDER BY N_avg DESC;

-- Query 4: Crops with Balanced NPK (all moderate: 40-80)
SELECT 
    crop,
    ROUND(N_avg, 2) as nitrogen,
    ROUND(P_avg, 2) as phosphorus,
    ROUND(K_avg, 2) as potassium
FROM crop_requirements
WHERE N_avg BETWEEN 40 AND 80
    AND P_avg BETWEEN 40 AND 80
    AND K_avg BETWEEN 40 AND 80
ORDER BY crop;

-- Query 5: Top 5 Phosphorus Consumers
SELECT 
    crop,
    ROUND(P_avg, 2) as avg_phosphorus,
    ROUND(P_max, 2) as max_phosphorus
FROM crop_requirements
ORDER BY P_avg DESC
LIMIT 5;

-- Query 6: Top 5 Potassium Consumers
SELECT 
    crop,
    ROUND(K_avg, 2) as avg_potassium,
    ROUND(K_max, 2) as max_potassium
FROM crop_requirements
ORDER BY K_avg DESC
LIMIT 5;

-- CLIMATE ANALYSIS QUERIES

-- Query 7: High Temperature Tolerant Crops (temp > 30°C)
SELECT 
    crop,
    ROUND(temp_avg, 2) as avg_temperature,
    ROUND(temp_max, 2) as max_temperature,
    ROUND(rainfall_avg, 2) as avg_rainfall
FROM crop_requirements
WHERE temp_avg > 30
ORDER BY temp_avg DESC;

-- Query 8: Cool Climate Crops (temp < 20°C)
SELECT 
    crop,
    ROUND(temp_avg, 2) as avg_temperature,
    ROUND(temp_min, 2) as min_temperature
FROM crop_requirements
WHERE temp_avg < 20
ORDER BY temp_avg ASC;

-- Query 9: High Rainfall Crops (rainfall > 200mm)
SELECT 
    crop,
    ROUND(rainfall_avg, 2) as avg_rainfall_mm,
    ROUND(rainfall_max, 2) as max_rainfall,
    ROUND(humidity_avg, 2) as avg_humidity
FROM crop_requirements
WHERE rainfall_avg > 200
ORDER BY rainfall_avg DESC;

-- Query 10: Drought Tolerant Crops (rainfall < 80mm)
SELECT 
    crop,
    ROUND(rainfall_avg, 2) as avg_rainfall_mm,
    ROUND(rainfall_min, 2) as min_rainfall,
    ROUND(temp_avg, 2) as avg_temperature
FROM crop_requirements
WHERE rainfall_avg < 80
ORDER BY rainfall_avg ASC;


-- SOIL pH ANALYSIS QUERIES

-- Query 11: Crops for Acidic Soil (pH < 6)
SELECT 
    crop,
    ROUND(ph_avg, 2) as avg_ph,
    ROUND(ph_min, 2) as min_ph,
    ROUND(ph_max, 2) as max_ph
FROM crop_requirements
WHERE ph_avg < 6
ORDER BY ph_avg ASC;

-- Query 12: Crops for Alkaline Soil (pH > 7.5)
SELECT 
    crop,
    ROUND(ph_avg, 2) as avg_ph,
    ROUND(ph_max, 2) as max_ph
FROM crop_requirements
WHERE ph_avg > 7.5
ORDER BY ph_avg DESC;

-- Query 13: pH Flexible Crops (pH range > 2)
SELECT 
    crop,
    ROUND(ph_avg, 2) as avg_ph,
    ROUND(ph_max - ph_min, 2) as ph_range,
    ROUND(ph_min, 2) as min_ph,
    ROUND(ph_max, 2) as max_ph
FROM crop_requirements
WHERE (ph_max - ph_min) > 2
ORDER BY (ph_max - ph_min) DESC;

-- CROP CATEGORIZATION QUERIES

-- Query 14: Categorize Crops by Nitrogen Requirements
SELECT 
    crop,
    ROUND(N_avg, 2) as avg_nitrogen,
    CASE 
        WHEN N_avg < 40 THEN 'Low N (Nitrogen Fixers)'
        WHEN N_avg < 80 THEN 'Medium N'
        ELSE 'High N (Heavy Feeders)'
    END as nitrogen_category
FROM crop_requirements
ORDER BY N_avg DESC;

-- Query 15: Categorize by Climate Type
SELECT 
    crop,
    ROUND(temp_avg, 2) as avg_temp,
    ROUND(rainfall_avg, 2) as avg_rainfall,
    CASE 
        WHEN temp_avg < 20 AND rainfall_avg < 100 THEN 'Cool & Dry'
        WHEN temp_avg < 20 AND rainfall_avg >= 100 THEN 'Cool & Wet'
        WHEN temp_avg >= 20 AND rainfall_avg < 100 THEN 'Warm & Dry'
        ELSE 'Warm & Wet (Tropical)'
    END as climate_type
FROM crop_requirements
ORDER BY climate_type, crop;

-- Query 16: Categorize by Water Requirements
SELECT 
    crop,
    ROUND(rainfall_avg, 2) as avg_rainfall,
    CASE 
        WHEN rainfall_avg < 60 THEN 'Very Low (Drought Tolerant)'
        WHEN rainfall_avg < 120 THEN 'Low to Moderate'
        WHEN rainfall_avg < 180 THEN 'Moderate to High'
        ELSE 'Very High (Water Intensive)'
    END as water_requirement
FROM crop_requirements
ORDER BY rainfall_avg DESC;


-- CROP SUITABILITY CHECK QUERIES

-- Query 17: Find Suitable Crops for Given Conditions
-- Example: N=90, P=42, K=43, temp=21, humidity=82, ph=6.5, rainfall=202
SELECT 
    crop,
    CASE 
        WHEN 90 BETWEEN N_min AND N_max AND
             42 BETWEEN P_min AND P_max AND
             43 BETWEEN K_min AND K_max AND
             21 BETWEEN temp_min AND temp_max AND
             82 BETWEEN humidity_min AND humidity_max AND
             6.5 BETWEEN ph_min AND ph_max AND
             202 BETWEEN rainfall_min AND rainfall_max
        THEN '✓ Suitable'
        ELSE '✗ Not Suitable'
    END as suitability,
    ROUND(N_avg, 2) as ideal_N,
    ROUND(P_avg, 2) as ideal_P,
    ROUND(K_avg, 2) as ideal_K,
    ROUND(temp_avg, 2) as ideal_temp
FROM crop_requirements
WHERE suitability = '✓ Suitable'
ORDER BY crop;


-- CROP SIMILARITY QUERIES

-- Query 18: Find Similar Crops to Rice
-- Based on NPK requirements
WITH rice_requirements AS (
    SELECT N_avg, P_avg, K_avg, temp_avg, rainfall_avg
    FROM crop_requirements
    WHERE crop = 'rice'
)
SELECT 
    cr.crop,
    ROUND(ABS(cr.N_avg - rr.N_avg), 2) as N_difference,
    ROUND(ABS(cr.P_avg - rr.P_avg), 2) as P_difference,
    ROUND(ABS(cr.K_avg - rr.K_avg), 2) as K_difference,
    ROUND(
        ABS(cr.N_avg - rr.N_avg) + 
        ABS(cr.P_avg - rr.P_avg) + 
        ABS(cr.K_avg - rr.K_avg), 2
    ) as total_difference
FROM crop_requirements cr, rice_requirements rr
WHERE cr.crop != 'rice'
ORDER BY total_difference ASC
LIMIT 5;


-- STATISTICAL SUMMARY QUERIES

-- Query 19: Overall Feature Statistics
SELECT 
    'Nitrogen' as feature,
    ROUND(AVG(N_avg), 2) as overall_avg,
    ROUND(MIN(N_min), 2) as overall_min,
    ROUND(MAX(N_max), 2) as overall_max
FROM crop_requirements
UNION ALL
SELECT 
    'Phosphorus',
    ROUND(AVG(P_avg), 2),
    ROUND(MIN(P_min), 2),
    ROUND(MAX(P_max), 2)
FROM crop_requirements
UNION ALL
SELECT 
    'Potassium',
    ROUND(AVG(K_avg), 2),
    ROUND(MIN(K_min), 2),
    ROUND(MAX(K_max), 2)
FROM crop_requirements
UNION ALL
SELECT 
    'Temperature',
    ROUND(AVG(temp_avg), 2),
    ROUND(MIN(temp_min), 2),
    ROUND(MAX(temp_max), 2)
FROM crop_requirements
UNION ALL
SELECT 
    'Humidity',
    ROUND(AVG(humidity_avg), 2),
    ROUND(MIN(humidity_min), 2),
    ROUND(MAX(humidity_max), 2)
FROM crop_requirements
UNION ALL
SELECT 
    'pH',
    ROUND(AVG(ph_avg), 2),
    ROUND(MIN(ph_min), 2),
    ROUND(MAX(ph_max), 2)
FROM crop_requirements
UNION ALL
SELECT 
    'Rainfall',
    ROUND(AVG(rainfall_avg), 2),
    ROUND(MIN(rainfall_min), 2),
    ROUND(MAX(rainfall_max), 2)
FROM crop_requirements;

-- Query 20: Crop Count by Category
SELECT 
    CASE 
        WHEN N_avg < 40 THEN 'Low N Crops'
        WHEN N_avg < 80 THEN 'Medium N Crops'
        ELSE 'High N Crops'
    END as category,
    COUNT(*) as crop_count,
    GROUP_CONCAT(crop, ', ') as crops
FROM crop_requirements
GROUP BY category
ORDER BY crop_count DESC;