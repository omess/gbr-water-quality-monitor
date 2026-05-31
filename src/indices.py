# src/indices.py
import ee

def calculate_marine_indices(image: ee.Image) -> ee.Image:
    """
    Performs server-side band math calculations to isolate open water
    and quantify turbidity and chlorophyll-a metrics.
    """
    # Calculate MNDWI: (Green - SWIR) / (Green + SWIR)
    mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    
    # Calculate NDTI (Turbidity): (Red - Green) / (Red + Green)
    ndti = image.normalizedDifference(['B4', 'B3']).rename('NDTI')
    
    # Chlorophyll-a empirical proxy: Blue / RedEdge1 (B2 / B5)
    # Fertilized runoff triggers algal blooms, shifting the blue-to-red edge response ratio
    chlorophyll = image.expression('B2 / B5', {
        'B2': image.select('B2'),
        'B5': image.select('B5')
    }).rename('CHLOROPHYLL_PROXY')
    
    # Create a dynamic water mask where MNDWI is positive (> 0.1)
    # This excludes beaches, mangrove forests, and agricultural soil noise
    water_mask = mndwi.gt(0.1)
    
    # Add computed indices back as bands and mask out all land masses
    return image.addBands([mndwi, ndti, chlorophyll]).updateMask(water_mask)