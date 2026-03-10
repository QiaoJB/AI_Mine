# UAV & Remote Sensing Based Mineral Prospectivity Prediction

This project implements a **remote sensing–driven mineral prospectivity prediction framework** using multi-source geospatial data and machine learning methods.

The workflow integrates:

* **Satellite remote sensing features**
* **Geological structure indicators**
* **Lithology information**
* **Machine learning / unsupervised anomaly detection**

The system supports multiple mineral types including:

* Gold
* Copper
* Iron
* Coal

and provides both **unsupervised exploration** and **supervised prediction** pipelines.

---

# 1. Project Overview

Mineral exploration traditionally relies on field surveys and geological interpretation, which are costly and time-consuming.

This project aims to:

* Automatically extract **alteration indicators** from satellite imagery
* Integrate **lithology, structure, and spectral features**
* Apply **machine learning methods** to identify mineral prospectivity zones

The system is designed for **large-scale remote sensing based mineral prediction**.

---

# 2. Data Sources

The framework integrates multiple geospatial datasets:

### Remote Sensing Data

Sentinel-2 multispectral imagery

Used bands:

* B02 (Blue)
* B03 (Green)
* B04 (Red)
* B08 (NIR)
* B11 (SWIR1)
* B12 (SWIR2)

These bands enable extraction of mineral alteration indicators.

---

### DEM Data

Digital Elevation Model used for structural features:

* slope
* curvature
* terrain gradient
* fault density

---

### Geological Data

Lithology map is converted into **one-hot encoded features** to provide geological background information.

---

# 3. Feature Engineering

The system constructs a **feature cube** integrating three types of information.

```
Feature Cube
│
├── Lithology features
├── Alteration spectral features
└── Structural features
```

---

## 3.1 Lithology Features

Lithology categories are encoded using **one-hot vectors**.

Example:

```
granite
basalt
sandstone
limestone
...
```

This allows machine learning models to utilize geological background information.

---

## 3.2 Alteration Spectral Indices

Different minerals produce characteristic spectral responses.

The framework extracts mineral alteration indices from Sentinel-2 imagery.

Examples:

* Fe oxide index
* Hematite index
* Al-OH index
* Chlorite index
* Silica proxy

These indices represent hydrothermal alteration associated with mineralization.

---

## 3.3 Environmental Suppression Factors

Vegetation and surface moisture can obscure spectral signals.

Therefore, the following indices are used as **suppression factors rather than features**:

* NDVI (vegetation)
* Moisture index

These are used to reduce false anomalies:

```
alteration_corrected = alteration * (1 - vegetation) * (1 - moisture)
```

---

## 3.4 Structural Features

Mineral deposits are often structurally controlled.

DEM-derived features include:

* slope
* curvature
* DEM gradient
* lineament / fault density

Structural density maps significantly improve mineral prospectivity prediction.

---

# 4. Supported Mineral Types

Different minerals require different spectral indicators.

The project provides configurable mineral-specific settings.

### Gold

Typical hydrothermal alteration:

* Fe oxide
* Al-OH
* Chlorite
* Silicification

```
fe_oxide_nd
hematite_nd
al_oh_nd
chlorite_nd
silica_proxy
```

---

### Copper

Porphyry copper alteration:

* Fe oxide
* Sericite
* Chlorite
* Clay alteration

```
fe_oxide_nd
hematite_nd
al_oh_nd
chlorite_nd
clay_ratio
```

---

### Iron

Iron ore exploration focuses on iron oxide anomalies.

```
fe_oxide_nd
hematite_nd
redness
darkness
```

---

### Coal

Coal deposits are characterized by low reflectance.

```
darkness
silica_proxy
```

---

# 5. Machine Learning Models

The framework supports multiple prediction strategies depending on sample availability.

---

## 5.1 No Sample Scenario

Unsupervised anomaly detection:

* Isolation Forest
* Local Outlier Factor

Used to identify spectral anomalies related to mineralization.

---

## 5.2 Small Sample Scenario

Semi-supervised methods:

* Gaussian Mixture Model
* Isolation Forest

Suitable for early exploration stages with limited known deposits.

---

## 5.3 Sufficient Samples

Supervised learning:

* Random Forest

Used to produce **mineral prospectivity maps**.

---

# 6. Prediction Outputs

The framework produces raster prediction maps:

```
gold_unsupervised_pure_feature.tif
gold_prediction_small_samples.tif
gold_rf_prediction_optimized.tif
```

These maps represent mineral prospectivity probabilities.

---

# 7. Visualization Tools

Interactive visualization tools are included.

Features:

* Overlay prediction results on satellite imagery
* Visualize feature channels
* Adjust transparency using sliders
* Display sample points

---

# 8. Project Structure

```
mine_predict/
│
├── configs
│   ├── gold.yaml
│   ├── copper.yaml
│   ├── iron.yaml
│   └── coal.yaml
│
├── preprocess
│   ├── alteration_features.py
│   ├── lithology_features.py
│   └── structural_features.py
│
├── models
│   ├── unsupervised_if_model.py
│   ├── small_sample_gmm_if_model.py
│   └── supervised_rf_model.py
│
├── visualization
│   └── visual_feature_interactive.py
│
└── data
```

---

# 9. Running the Pipeline

Example workflow:

### Step 1 – Generate alteration features

```
python preprocess/alteration_features.py
```

---

### Step 2 – Build feature cube

```
python preprocess/build_feature_cube.py
```

---

### Step 3 – Run prediction models

Unsupervised:

```
python models/unsupervised_if_model.py
```

Small samples:

```
python models/small_sample_gmm_if_model.py
```

Supervised:

```
python models/supervised_rf_model.py
```

---

# 10. Applications

This framework can be applied to:

* regional mineral exploration
* mineral prospectivity mapping
* geological remote sensing analysis
* UAV remote sensing integration

---

# 11. Future Work

Planned improvements:

* integration with UAV hyperspectral data
* multimodal remote sensing fusion
* deep learning based mineral detection
* 3D geological visualization

---

# License

Apache-2.0 license
