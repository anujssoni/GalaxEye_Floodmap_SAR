# Sentinel-1 SAR Flood Mapping – Ganga River, Patna (Sept 2024)

## 📌 Overview
This project uses **multi-temporal Earth Observation (EO) data** to assess the impact of the September 2024 flood event along the **Ganga River, Patna, Bihar**.  
The goal is to **detect and map flood-affected areas** by comparing Sentinel-1 imagery acquired before and after the event.  
The script is designed to produce georeferenced flood and water masks, calculate inundated areas, and deliver clear, interpretable outputs for both technical and non-technical stakeholders.

---

## 🗂 Data Used
- **Pre-flood Sentinel-1 GRD imagery** (VH/VV bands) – preprocessed (calibration, speckle filtering) and stored as GeoTIFF.  
- **Post-flood Sentinel-1 GRD imagery** (VH/VV bands) – same preprocessing pipeline.  
- **AOI shapefile** defining the Patna study area (optional).  

---

## ⚙️ Methodology
1. **Data Loading & AOI Clipping** – Reads VH bands for pre- and post-flood dates, optionally clips to AOI.  
2. **Unit Detection & Conversion** – Automatically detects if imagery is in dB or linear scale, converts as needed.  
3. **Speckle Filtering** – Applies **directional Refined-Lee filter** in memory-safe tiles to preserve edges.  
4. **Flood & Water Classification** – Uses dB thresholds to classify:
   - **Water Mask** (pre-existing water bodies)  
   - **Flood Mask** (newly inundated areas)  
5. **Morphological Cleaning** – Removes noise via median filtering, binary opening, and small-object removal.  
6. **Area Calculation** – Computes flood and water areas in hectares.  
7. **Accuracy Assessment** – Generates a proxy confusion matrix (pre-water vs post-flood masks).  
8. **Outputs**:
   - `flood_mask.tif`
   - `water_mask.tif`
   - `flood_water_classified.tif` (combined)
   - Diagnostic plots

---

## 📍 Study Area
- **Location:** Ganga River, Patna, Bihar, India  
- **Event Date:** September 2024 flood

---

## 📊 Limitations & Recommendations
- Thresholds may need tuning for different locations/dates.
- No ground truth data used – results are indicative, not validated.
- Further improvements: terrain correction, vectorized outputs, integration with optical imagery.

---

## ▶️ Usage
1. Install required Python packages:
   ```bash
   pip install numpy rasterio geopandas scipy scikit-image matplotlib scikit-learn

---

## 📊 Methodology Flowchart
![Methodology_Flowchart]("image/Methodology_flowchart.png")

## 🛰️ Sample Flood Mapping Result
![Pre_Flood_Event](images/Pre_flood_event.png)
![Post_Flood_Event](images/Post_flood_event.png)
![Zone_classification_result](images/Zone_classification.png)
