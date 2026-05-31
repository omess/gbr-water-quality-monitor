# src/ml_pipeline.py
"""
Geospatial Machine Learning Pipeline.
Constructs predictive matrices and optimizes an XGBoost Regressor model.
"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def assemble_synthetic_environmental_matrix(samples: int = 400) -> pd.DataFrame:
    """
    Simulates high-fidelity geospatial data mimicking the Burdekin Catchment ecosystem.
    Models the direct relationship between upstream land management and downstream plume turbidity.
    """
    np.random.seed(42)
    
    # Feature 1: Upstream land area dedicated to intensive agriculture/grazing (%)
    upstream_agri_pct = np.random.uniform(10.0, 95.0, samples)
    
    # Feature 2: 3-day cumulative localized catchment rainfall (mm)
    rainfall_3d_mm = np.random.uniform(0.0, 250.0, samples)
    
    # Feature 3: Distance from the coastal discharge mouth node (km)
    distance_mouth_km = np.random.uniform(1.0, 55.0, samples)
    
    # Target Equation: Simulates real-world coastal hydrodynamic decay behavior
    # Heavy rainfall over heavily cleared agricultural blocks spikes down-gradient sediment transport.
    base_turbidity = (
        (0.50 * upstream_agri_pct) + 
        (0.80 * rainfall_3d_mm) - 
        (0.40 * distance_mouth_km)
    )
    
    # Inject Gaussian noise to simulate wind-driven resuspension anomalies
    environmental_noise = np.random.normal(0, 7.5, samples)
    downstream_turbidity = np.clip(base_turbidity + environmental_noise, a_min=0.2, a_max=None)
    
    return pd.DataFrame({
        'upstream_agri_pct': upstream_agri_pct,
        'rainfall_3d_mm': rainfall_3d_mm,
        'distance_mouth_km': distance_mouth_km,
        'downstream_turbidity': downstream_turbidity
    })

def run_pipeline_training() -> None:
    """Executes dataset splits, optimizes the structural tree ensemble, and saves the asset."""
    print("\n--- Starting Machine Learning Model Training ---")
    df = assemble_synthetic_environmental_matrix()
    
    # Split features and target labels
    X = df[['upstream_agri_pct', 'rainfall_3d_mm', 'distance_mouth_km']]
    y = df['downstream_turbidity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize optimized gradient-boosted tree structure
    model = XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Compute operational validation metrics
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Optimization Status: SUCCESS")
    print(f"Training R² Score:  {model.score(X_train, y_train):.3f}")
    print(f"Validation R² Score: {r2:.3f}")
    print(f"Model Root Mean Squared Error: {rmse:.2f} NTU")
    
    # Secure storage of serialization file
    output_dir = os.path.join('data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'xgboost_sediment_model.pkl')
    
    joblib.dump(model, model_path)
    print(f"Serialized model saved successfully to: {model_path}\n")

if __name__ == "__main__":
    run_pipeline_training()