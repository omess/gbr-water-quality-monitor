# src/ee_utils.py
import ee
import pandas as pd
from typing import Dict, Any
from src.config import BURDEKIN_ROI, CLOUD_THRESHOLD

def initialize_gee() -> None:
    """Initializes Google Earth Engine API secure session."""
    try:
        ee.Initialize(project='your-gee-project-id') # Replace with your active GEE project ID
    except Exception as e:
        print("GEE Initialization failed. Ensure you have authenticated via 'earthengine authenticate'")
        raise e

def mask_sentinel_clouds(image: ee.Image) -> ee.Image:
    """Masks clouds in Sentinel-2 images using the QA60 bitmask band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])

def compute_water_quality_indices(image: ee.Image) -> ee.Image:
    """
    Computes professional-grade Marine Water Quality Indices on server-side GEE.
    B2=Blue, B3=Green, B4=Red, B5=RedEdge1, B8=NIR, B11=SWIR
    """
    # 1. MNDWI for clean water-masking
    mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    
    # 2. NDTI for Turbidity Monitoring
    ndti = image.normalizedDifference(['B4', 'B3']).rename('NDTI')
    
    # 3. Turbidity Proxy (Red/Green Ratio)
    turbidity = image.expression('B4 / B3', {
        'B4': image.select('B4'),
        'B3': image.select('B3')
    }).rename('TURBIDITY_PROXY')
    
    # 4. Chlorophyll-a Proxy (Blue/Red-Edge Ratio)
    chlorophyll = image.expression('B2 / B5', {
        'B2': image.select('B2'),
        'B5': image.select('B5')
    }).rename('CHLOROPHYLL_PROXY')
    
    # Isolate open water bodies to exclude terrestrial vegetation and soil noise
    water_mask = mndwi.gt(0.1)
    
    return image.addBands([mndwi, ndti, turbidity, chlorophyll]).updateMask(water_mask)

def fetch_event_composites(start_date: str, end_date: str) -> ee.Image:
    """Fetches a cloud-free, median-reduced water quality composite for a given time window."""
    roi = ee.Geometry.Polygon(BURDEKIN_ROI['coordinates'])
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD))
                  .map(mask_sentinel_clouds)
                  .map(compute_water_quality_indices))
    
    # Spatial median aggregation across the timeframe to extract a noise-free image
    return collection.median().clip(roi)

def generate_time_series_metrics() -> pd.DataFrame:
    """Extracts aggregated historical metrics to feed the Streamlit ESG Scorecard."""
    roi = ee.Geometry.Polygon(BURDEKIN_ROI['coordinates'])
    
    # Gather monthly composites across a critical monitoring year
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi)
                  .filterDate('2019-01-01', '2019-12-31')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD))
                  .map(mask_sentinel_clouds)
                  .map(compute_water_quality_indices))
    
    def calculate_spatial_mean(img):
        date = img.date().format('YYYY-MM-DD')
        mean_vals = img.select(['NDTI', 'TURBIDITY_PROXY', 'CHLOROPHYLL_PROXY']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=60,
            maxPixels=1e8
        )
        return ee.Feature(None, {'date': date}).set(mean_vals)

    # Convert server-side feature extraction to client-side Pandas DataFrame
    features = collection.map(calculate_spatial_mean).getInfo()
    
    data_list = []
    for feat in features['features']:
        props = feat['properties']
        if 'NDTI' in props: # Filter out scenes with complete cloud gaps
            data_list.append(props)
            
    df = pd.DataFrame(data_list)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')