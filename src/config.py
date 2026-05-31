# src/config.py

# Bounding box for Burdekin River Mouth / Upstart Bay, Queensland
BURDEKIN_ROI = {
    "type": "Polygon",
    "coordinates": [[
        [147.20, -19.90],
        [147.90, -19.90],
        [147.90, -19.25],
        [147.20, -19.25],
        [147.20, -19.90]
    ]]
}

# Date Anchors for Event-Based Verification (2019 Extreme Queensland Floods)
EVENTS = {
    "dry_baseline": {"start": "2018-09-01", "end": "2018-11-30", "label": "Dry Season Baseline"},
    "flood_peak": {"start": "2019-02-10", "end": "2019-03-05", "label": "Extreme Flood Dispersal"},
    "post_recovery": {"start": "2019-06-01", "end": "2019-08-31", "label": "Post-Event Recovery"}
}

CLOUD_THRESHOLD = 20  # Max acceptable cloud coverage percentage