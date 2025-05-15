import os
import torch
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from model.predict_utils import (
    load_model,
    predict_image,
    save_to_geojson,
    calculate_area_and_energy
)

# ===== CONFIGURATION =====
UPLOAD_FOLDER = 'data/uploads'
RESULT_FOLDER = 'data/results'
MODEL_PATH = 'model/segformer_model.pth'
IRRADIANCE_CSV = 'data/irradiance_data.csv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== FLASK APP SETUP =====
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once at app startup
model = load_model(MODEL_PATH, DEVICE)


# ===== ROUTES =====
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if file part is in the request
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No image selected", 400

    # Secure and save uploaded file
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # Run segmentation and extract polygons
    polygons = predict_image(image_path, model, DEVICE)

    # Save predicted polygons to GeoJSON
    geojson_path = os.path.join(RESULT_FOLDER, filename.replace('.tif', '.geojson'))
    save_to_geojson(polygons, geojson_path)

    # Calculate area and solar energy output
    result_csv_path = os.path.join(RESULT_FOLDER, filename.replace('.tif', '_results.csv'))
    results = calculate_area_and_energy(
        polygon_geojson_path=geojson_path,
        reference_image_path=image_path,
        irradiance_csv_path=IRRADIANCE_CSV,
        output_csv_path=result_csv_path
    )

    # Render results on web page
    return render_template(
        'result.html',
        area=round(results["area_m2"], 2),
        irradiance=round(results["irradiance"], 2),
        energy=round(results["energy_kWh"], 2),
        geojson_file=os.path.basename(geojson_path),
        csv_file=os.path.basename(result_csv_path)
    )


# ===== MAIN ENTRY POINT =====
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True)
