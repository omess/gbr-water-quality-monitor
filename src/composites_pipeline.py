# src/composites_pipeline.py
import ee
from src.config import BURDEKIN_ROI, CLOUD_FILTER_THRESHOLD
from src.ee_utils import apply_sentinel_cloud_mask
from src.indices import calculate_marine_indices

def generate_epoch_mosaic(start_date: str, end_date: str) -> ee.Image:
    """
    Filters, masks, and computes water metrics over a target time window,
    returning a single median-reduced, cloud-free structural image.
    """
    # Instantiate Earth Engine geographic geometry
    roi_geometry = ee.Geometry.Polygon(BURDEKIN_ROI['coordinates'])
    
    # Build collection pipeline query
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi_geometry)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER_THRESHOLD))
                  .map(apply_sentinel_cloud_mask)
                  .map(calculate_marine_indices))
    
    # Use a median reducer to eliminate lingering cloud artifacts or moving boats
    median_mosaic = collection.median().clip(roi_geometry)
    return median_mosaic