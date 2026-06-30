# Great Barrier Reef Catchment Sediment & Water Quality Monitoring Platform

### An Enterprise Environmental Intelligence Application pairing Sentinel-2 MultiSpectral Imagery with Gradient-Boosted Machine Learning (XGBoost)

---

## 📌 Executive Summary
Terrestrial agricultural runoff from severe monsoonal events is a primary driver of water quality degradation across the Great Barrier Reef (GBR) lagoon. Suspended sediments attenuate light penetration, choking out seagrass meadows, while high nutrient concentrations trigger massive, destructive macroalgal blooms. 

This platform provides an automated, scalable approach to remote water quality auditing. By leveraging **Google Earth Engine (GEE)** for cloud-based, server-side raster algebra and **XGBoost Regression** for downstream predictive modeling, this tool connects upstream land-management indicators to downstream marine environmental risk.

---

## 🛠️ System Architecture & Workflow



The application operates across three distinct computational layers:
1. **Data Ingestion & Atmospheric Core (GEE API):** Queries Copernicus Sentinel-2 Level-2A surface reflectance data over the Burdekin River mouth (Upstart Bay). Applies cloud/cirrus bitmasking via the `QA60` band.
2. **Spectral Index Calculation Engine:** Restricts calculations to open water bodies via the Modified Normalized Difference Water Index (MNDWI) to remove terrestrial noise. Computes the Normalized Difference Turbidity Index (NDTI) and custom Chlorophyll-a empirical proxies.
3. **Machine Learning Predictive Layer:** Loads a serialized, high-performance XGBoost Regressor trained on regional environmental parameters (upstream agricultural clearing density, spatial decay metrics, and 3-day cumulative precipitation inputs) to evaluate and alert down-gradient plume severity in Real-Time ($R^2 = 0.976$).

---

## 📁 Repository Directory Blueprint
```text
gbr-water-quality-monitor/
├── .github/workflows/        # Automated deployment definitions
├── app/
│   └── main.py               # Streamlit Multi-page Interactive Interface
├── data/
│   ├── raw/                  # Spatial boundaries & coordinate arrays
│   └── processed/            # Serialized XGBoost ML Model (.pkl)
├── src/
│   ├── __init__.py
│   ├── config.py             # Target spatial coordinates and active Project ID
│   ├── ee_utils.py           # Earth Engine extraction and cloud masking engines
│   └── ml_pipeline.py        # XGBoost training matrix and validation logic
├── requirements.txt          # Explicitly pinned library dependencies
└── README.md                 # Executive portfolio documentation