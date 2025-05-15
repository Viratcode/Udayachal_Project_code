# predict_utils.py

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from shapely.geometry import shape
from rasterio.features import shapes
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import geopandas as gpd
from shapely.ops import unary_union
from skimage.measure import find_contours


# -----------------------------------
# Load SegFormer model
# -----------------------------------
def load_model(model_path, device):
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# -----------------------------------
# Predict mask and extract polygons
# -----------------------------------
def predict_image(image_path, model, device):
    # Load image (assumed RGB) using rasterio
    with rasterio.open(image_path) as src:
        img_array = src.read([1, 2, 3])  # Extract RGB bands
        img_array = np.moveaxis(img_array, 0, -1)  # Convert to (H, W, C)

    # Normalize and prepare for inference
    input_image = Image.fromarray(img_array.astype(np.uint8))
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b0")
    inputs = feature_extractor(images=input_image, return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get predicted class per pixel
    prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

    # Convert mask to polygons
    polygons = convert_to_polygons(prediction)

    return polygons


# -----------------------------------
# Convert binary mask to valid polygons
# -----------------------------------
def convert_to_polygons(mask):
    mask = mask.astype(np.uint8)
    results = (
        {'geometry': shape(geom), 'value': val}
        for geom, val in shapes(mask, mask=mask > 0)
        if val == 1  # Only class 1 (buildings)
    )
    polygons = [r["geometry"] for r in results if shape(r["geometry"]).is_valid]
    return polygons


# -----------------------------------
# Save polygon GeoDataFrame to GeoJSON
# -----------------------------------
def save_to_geojson(polygons, geojson_path, crs="EPSG:4326"):
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf.to_file(geojson_path, driver="GeoJSON")


# -----------------------------------
# Calculate area and solar energy potential
# -----------------------------------
def calculate_area_and_energy(
    polygon_geojson_path: str,
    reference_image_path: str,
    irradiance_csv_path: str,
    output_csv_path: str
):
    """
    Calculates the total rooftop area and solar energy potential.

    Parameters:
    - polygon_geojson_path (str): Path to predicted building polygons (GeoJSON).
    - reference_image_path (str): Path to original georeferenced input TIFF (for CRS info).
    - irradiance_csv_path (str): Path to CSV with irradiance data (from NASA POWER).
    - output_csv_path (str): Output path to save results (CSV).

    Returns:
    - dict with area (m²), irradiance (kWh/m²/year), and total energy (kWh/year).
    """

    # Load predicted building polygons
    gdf = gpd.read_file(polygon_geojson_path)

    # Get CRS from original georeferenced image
    with rasterio.open(reference_image_path) as src:
        transform = src.transform
        crs = src.crs

    # Reproject polygons to UTM zone for accurate area calculations
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    # Compute area in square meters
    gdf["area_m2"] = gdf.geometry.area

    # Determine center lat/lon of polygon bounding box
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    # Load solar irradiance data from CSV
    irradiance_df = pd.read_csv(irradiance_csv_path, skiprows=9)
    print("CSV Columns:", irradiance_df.columns.tolist())  # <-- Add this

    # Find the record closest to center of bounding box
    irradiance_df["distance"] = ((irradiance_df["LAT"] - center_lat) ** 2 +
                                 (irradiance_df["LON"] - center_lon) ** 2)
    closest = irradiance_df.loc[irradiance_df["distance"].idxmin()]
    ann_irradiance = closest["ANN"]  # Annual average daily irradiance (kWh/m²/day)

    # Total predicted area
    total_area_m2 = gdf["area_m2"].sum()

    # Annual solar energy potential = irradiance * area
    total_energy_kWh = total_area_m2 * ann_irradiance

    # Save result as CSV
    result_df = pd.DataFrame({
        "center_lat": [center_lat],
        "center_lon": [center_lon],
        "total_area_m2": [total_area_m2],
        "irradiance_kWh_per_m2": [ann_irradiance],
        "total_energy_kWh": [total_energy_kWh]
    })
    result_df.to_csv(output_csv_path, index=False)

    return {
        "area_m2": total_area_m2,
        "irradiance": ann_irradiance,
        "energy_kWh": total_energy_kWh
    }
