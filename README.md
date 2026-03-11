# AI-Based Mineral Prospectivity Prediction and 3D Geological Modeling

This project implements an **integrated mineral exploration framework** combining:

* Satellite remote sensing analysis
* Geological feature extraction
* Machine learning–based prospectivity prediction
* Drillhole-based **3D geological modeling**

The system is designed for **regional mineral exploration and geological interpretation**, supporting both **surface spectral analysis** and **subsurface geological modeling**.

Supported mineral types include:

* Gold
* Copper
* Iron
* Coal

The framework integrates **remote sensing features, geological structure information, lithology data, and drillhole constraints** to generate **mineral prospectivity maps and 3D geological models**.

---

# 1. Project Overview

Mineral exploration traditionally relies on geological surveys, drilling, and manual interpretation. These processes are costly and time-consuming.

This project aims to provide an automated workflow that integrates:

* **Remote sensing alteration detection**
* **Geological structural analysis**
* **Machine learning–based anomaly detection**
* **Drillhole-driven 3D geological modeling**

The final outputs include:

* Mineral prospectivity maps
* Geological feature maps
* 3D stratigraphic models

---

# 2. Data Sources

The framework integrates multiple geospatial datasets.

## 2.1 Satellite Remote Sensing Data

Sentinel-2 multispectral imagery is used for spectral feature extraction.

Used bands:

* B02 – Blue
* B03 – Green
* B04 – Red
* B08 – Near Infrared
* B11 – SWIR1
* B12 – SWIR2

These bands allow extraction of mineral alteration indicators.

---

## 2.2 DEM Data

Digital Elevation Model data are used to derive structural indicators:

* slope
* curvature
* terrain gradient
* structural density

These features help characterize tectonic controls on mineralization.

---

## 2.3 Geological Data

Lithology maps are converted into **one-hot encoded feature layers**, providing geological background information for machine learning models.

---

## 2.4 Drillhole Data

Drillhole geological logs are used for **3D geological modeling**.

These data include:

* drillhole location (X, Y)
* elevation reference
* lithology layers
* stratigraphic thickness

Drillhole information provides **subsurface constraints** for geological modeling.

---

# 3. Feature Engineering

The system constructs a **multi-source feature cube** integrating three types of information.

```
Feature Cube
│
├── Lithology features
├── Spectral alteration features
└── Structural features
```

---

## 3.1 Lithology Features

Lithology categories are encoded using **one-hot vectors**.

Example lithology classes:

* granite
* basalt
* sandstone
* limestone
* shale

These categorical features allow machine learning models to incorporate geological context.

---

## 3.2 Alteration Spectral Indices

Hydrothermal alteration minerals produce characteristic spectral responses.

The framework extracts mineral alteration indices from Sentinel-2 imagery.

Examples:

* Fe oxide index
* Hematite index
* Al-OH index
* Chlorite index
* Silica proxy

These indicators are commonly associated with hydrothermal mineralization.

---

## 3.3 Environmental Suppression Factors

Vegetation and surface moisture can obscure spectral signals.

Therefore:

* **NDVI**
* **Moisture index**

are used as **suppression factors rather than predictive features**.

Example correction:

```
alteration_corrected =
    alteration * (1 - vegetation) * (1 - moisture)
```

This reduces false anomalies caused by vegetation or water.

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

# 4. Machine Learning Prediction Models

The framework supports multiple prediction strategies depending on sample availability.

---

## 4.1 No Sample Scenario

Unsupervised anomaly detection:

* Isolation Forest
* Local Outlier Factor (optional)

Used to identify spectral anomalies potentially related to mineralization.

---

## 4.2 Small Sample Scenario

Semi-supervised prediction:

* Gaussian Mixture Model
* Isolation Forest

Suitable when only a limited number of known deposits exist.

---

## 4.3 Sufficient Sample Scenario

Supervised learning:

* Random Forest classifier

This model produces **mineral prospectivity maps** based on labeled training data.

---

# 5. Drillhole-Based 3D Geological Modeling

The project includes a **3D geological modeling module** based on drillhole data.

Module location:

```
geo_modeling/geo_modeling.py
```

---

## 5.1 Modeling Workflow

The modeling process follows these steps:

### Step 1 – Drillhole Stratigraphic Construction

Using drillhole geological logs as primary constraints, stratigraphic structures are constructed for each individual drillhole.

All data are unified under a consistent:

* coordinate system
* elevation reference

---

### Step 2 – Regional Stratigraphic Correlation

Stratigraphic sequences are correlated across multiple drillholes to establish **regional stratigraphic relationships**.

This step ensures that layer continuity and stratigraphic order are preserved.

---

### Step 3 – Trend Surface Analysis

Trend surface analysis is applied to model the **overall spatial variation of stratigraphic top surfaces**.

This captures large-scale geological trends across the study area.

---

### Step 4 – Residual Analysis

Residual analysis is used to detect:

* local stratigraphic anomalies
* structural disturbances

Residual values indicate deviations from the regional trend surface.

---

### Step 5 – Residual Correction

Under the constraint of maintaining stratigraphic sequence relationships, corrections are applied using a **minimum residual principle**.

This ensures that:

* local anomalies are preserved
* stratigraphic continuity is maintained.

---

### Step 6 – Stratigraphic Surface Construction

Based on spatially distributed stratigraphic control points, individual **stratigraphic surface models** are constructed.

These surfaces represent the top boundaries of geological layers.

---

### Step 7 – 3D Geological Model Generation

The final step constructs a **3D geological model of the study area**, representing the spatial structure of major rock formations.

This model provides reliable subsurface information for:

* geological interpretation
* resource evaluation
* engineering applications.

---

# 6. Prediction Outputs

The system generates raster mineral prospectivity maps.

Examples:

```
gold_unsupervised_pure_feature.tif
gold_prediction_small_samples.tif
gold_rf_prediction_optimized.tif
```

These maps represent **mineral prospectivity probability distributions**.

---

# 7. Visualization Tools

Interactive visualization tools are included.

Features include:

* overlay prediction maps on satellite imagery
* feature channel visualization
* adjustable transparency
* drillhole and sample point display

This allows users to visually inspect relationships between geological features and prediction results.

---


# 8. Running the Pipeline

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

Unsupervised prediction:

```
python models/unsupervised_if_model.py
```

Small sample prediction:

```
python models/small_sample_gmm_if_model.py
```

Supervised prediction:

```
python models/supervised_rf_model.py
```

---

### Step 4 – Run 3D geological modeling

```
python geo_modeling/geo_modeling.py
```

---

# 9. Applications

This framework can be applied to:

* regional mineral exploration
* mineral prospectivity mapping
* geological remote sensing interpretation
* subsurface geological modeling
* geological engineering support

