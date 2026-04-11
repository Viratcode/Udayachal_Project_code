# Udayachal: AI-Powered Solar Potential Analysis Platform

**Udayachal** is a sophisticated geospatial web application that leverages deep learning and satellite imagery to automatically detect rooftop areas and calculate solar energy potential across Karnataka, India. The platform combines state-of-the-art semantic segmentation using SegFormer with geospatial processing and solar irradiance data to provide comprehensive solar energy assessments.

# LIVE Project DEMO 

link :- https://drive.google.com/file/d/1vCCyXJX6j3eKMwkKHMgj_3fku2AttXdM/view?usp=sharing

![Uploading image.png…]()







---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Deployment](#deployment)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Udayachal (meaning "sunrise" in Kannada) addresses the growing need for automated solar potential assessment in urban and rural areas of Karnataka. The platform enables users to:

1. **Upload GeoTIFF satellite imagery** for rooftop detection and solar analysis
2. **Draw custom regions of interest (ROI)** on an interactive map for on-demand analysis
3. **Automatically detect rooftop areas** using a fine-tuned SegFormer model
4. **Calculate solar energy potential** based on location-specific irradiance data
5. **Visualize results** with interactive maps, masks, and comprehensive metrics

The system processes satellite imagery at 0.6-meter resolution, ensuring accurate rooftop detection and precise solar energy calculations.

---

## Key Features

### Dual Input Modalities
- **GeoTIFF Upload**: Process high-resolution georeferenced satellite imagery
- **Interactive ROI Selection**: Draw rectangular regions on a satellite map for instant analysis

### AI-Powered Rooftop Detection
- **SegFormer Architecture**: State-of-the-art semantic segmentation model (MiT-B4 encoder)
- **Binary Classification**: Background vs. rooftop/building detection
- **Patch-based Processing**: Handles large images through intelligent tiling with overlap
- **Automatic Resampling**: Converts imagery to optimal 0.6m resolution for model inference

### Solar Potential Calculation
- **Location-Specific Irradiance**: Bilinear interpolation from NASA SSE solar database
- **Annual Energy Generation**: Calculates kWh/year based on rooftop area and solar data
- **Instantaneous Power**: Estimates peak power output in W/kW/MW
- **Panel Specifications**: Monocrystalline PERC panels (18% efficiency, 25-year lifespan)

### Geospatial Processing
- **CRS Detection & Conversion**: Automatic UTM zone calculation for accurate area measurements
- **GeoJSON Generation**: Converts prediction masks to georeferenced vector data
- **Karnataka Boundary Validation**: Ensures analysis is within supported regions
- **Google Satellite Tiles**: High-resolution basemap integration (zoom level 19)

### Visualization & Reporting
- **Interactive Leaflet Map**: Satellite imagery with drawing tools
- **Input/Mask Visualization**: Side-by-side comparison of original and processed images
- **Metrics Dashboard**: Real-time display of area, energy, and power statistics
- **Technical Specifications**: Detailed panel information and methodology

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Web)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ GeoTIFF      │  │ ROI Drawing  │  │ Results          │  │
│  │ Upload       │  │ Tool         │  │ Dashboard        │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Flask Backend (app.py)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ File Handler │  │ ROI          │  │ Response         │  │
│  │ & Validator  │  │ Processor    │  │ Formatter        │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ SegFormer    │  │ Geospatial   │  │ Irradiance   │
│ Model        │  │ Engine       │  │ Calculator   │
│ (PyTorch)    │  │ (Rasterio,   │  │ (Pandas,     │
│              │  │  GeoPandas)  │  │  Math)       │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## Technology Stack

### Backend Framework
- **Flask 2.0.1**: Lightweight WSGI web application framework
- **Python 3.8+**: Core programming language

### Deep Learning & AI
- **PyTorch 2.0.0**: Deep learning framework
- **Transformers 4.15.0**: Hugging Face library for SegFormer
- **SegFormer (nvidia/mit-b4)**: Pre-trained semantic segmentation model
- **OpenCV**: Image processing and augmentation
- **Albumentations**: Advanced image augmentation library

### Geospatial Processing
- **Rasterio 1.2.10**: Geospatial raster data I/O
- **GeoPandas 0.10.2**: Geospatial vector data operations
- **Shapely**: Computational geometry
- **PyProj 3.2.0**: Coordinate reference system transformations
- **Mercantile 1.2.1**: Web mercator tile calculations

### Data Processing & Visualization
- **NumPy 1.21.2**: Numerical computing
- **Pandas 1.3.3**: Data manipulation and analysis
- **Pillow 9.0.0**: Image processing
- **Matplotlib**: Plotting and visualization
- **Folium**: Interactive map generation

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **JavaScript (ES6+)**: Client-side logic
- **Leaflet 1.7.1**: Interactive map library
- **Leaflet.draw 1.0.4**: Drawing tools for maps
- **Google Fonts (Poppins)**: Typography
- **Font Awesome 6.0.0**: Icon library

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM (8GB+ recommended for model inference)
- GPU with CUDA support (optional, but recommended for faster processing)

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/Viratcode/Udayachal_Project_code.git
cd Udayachal_Project_code
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: If you encounter installation issues with specific packages, install them individually:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### 4. Download Pre-trained Model
The model should be placed in the `model/` directory:
```
model/
└── segformer_model.pth  # Your trained SegFormer weights
```

**Important**: The training code is maintained in a separate private branch. Contact the project maintainers for access to the trained model file.

#### 5. Verify Data Files
Ensure the irradiance data is present:
```
data/
└── irradiance_data.csv  # NASA SSE solar irradiance data
```

#### 6. Configure API Tokens
Open `app.py` and update the following configuration:
```python
MAPBOX_TOKEN = 'YOUR_MAPBOX_TOKEN'  # Replace with your Mapbox token (if using)
```

---

## Configuration

### Application Settings
Located in `app.py`:

```python
# File upload configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'tif', 'tiff'}
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max

# Model configuration
MODEL_PATH = 'model/segformer_model.pth'
IRRADIANCE_DATA_PATH = 'data/irradiance_data.csv'

# Processing parameters
TILE_SIZE = 256          # Google tile size in pixels
PATCH_SIZE = 512         # Image patch size for processing
OVERLAP = 64             # Patch overlap for seamless predictions

# Karnataka boundaries (bounding box)
KARNATAKA_BOUNDS = {
    'north': 18.4,
    'south': 11.5,
    'west': 74.0,
    'east': 78.6
}
```

### Solar Panel Parameters
```python
PANEL_EFFICIENCY = 0.18              # 18% efficiency (Monocrystalline PERC)
STANDARD_TEST_IRRADIANCE = 1000      # W/m² (STC conditions)
PANEL_LIFESPAN = 25                  # years
```

### Model Architecture
- **Base Model**: NVIDIA SegFormer (MiT-B4 encoder)
- **Number of Classes**: 2 (background, rooftop)
- **Input Size**: 256x256 pixels (training), variable (inference)
- **Output**: Semantic segmentation mask

---

## Usage

### Starting the Application

#### 1. Run the Flask Server
```bash
python app.py
```

The server will start at:
```
http://127.0.0.1:5000/
```

#### 2. Access the Web Interface
Open your browser and navigate to `http://127.0.0.1:5000/`

### Method 1: GeoTIFF Upload

1. **Prepare Your GeoTIFF**:
   - Ensure the file is in GeoTIFF format (.tif or .tiff)
   - File size must be under 32MB
   - Should contain satellite imagery of Karnataka region

2. **Upload & Analyze**:
   - Click "Choose File" in the upload section
   - Select your GeoTIFF file
   - Click "Analyze Image"
   - Wait for processing (typically 10-30 seconds depending on image size)

3. **View Results**:
   - Input satellite image visualization
   - Detected rooftop mask (binary)
   - Total rooftop area (m² or hectares)
   - Annual energy generation (kWh/MWh/GWh)
   - Instantaneous power capacity (W/kW/MW)

### Method 2: Region of Interest (ROI) Analysis

1. **Navigate to Location**:
   - Use the interactive map to zoom into your area of interest
   - The map defaults to Google Satellite imagery
   - You can use geolocation to center on your current position

2. **Draw Selection**:
   - Click the rectangle drawing tool
   - Draw a rectangular region on the map
   - The "Analyze Selected Region" button will become active

3. **Process ROI**:
   - Click "Analyze Selected Region"
   - The system will:
     - Download Google Satellite tiles for the selected area
     - Create a mosaic from multiple tiles
     - Process the mosaic through the SegFormer model
     - Calculate solar potential

4. **Review Results**:
   - Same metrics as GeoTIFF upload
   - Results displayed in the dashboard below the map

### Understanding the Results

#### Metrics Explained

| Metric | Description | Unit | Example |
|--------|-------------|------|---------|
| **Total Rooftop Area** | Combined area of all detected rooftops | m² / hectares | 2.45 hectares (24,500 m²) |
| **Annual Energy Generation** | Total solar energy produced per year | kWh / MWh / GWh | 1.23 MWh/year |
| **Instantaneous Power** | Peak power output at STC | W / kW / MW | 45.67 kW |
| **Annual Irradiance** | Solar energy received per m² | kWh/m²/year | 1,825 kWh/m²/year |

#### Calculation Formulas

**Instantaneous Power (W)**:
```
Power = Area (m²) × 1000 (W/m²) × 0.18 (efficiency)
```

**Annual Energy Generation (kWh)**:
```
Energy = Area (m²) × Annual Irradiance (kWh/m²/year) × 0.18 (efficiency)
```

---

## Model Training

The SegFormer model training code is maintained separately. Below is an overview of the training pipeline:

### Training Pipeline (`segformer_main.ipynb`)

#### 1. Dataset Preparation
```python
# Directory structure
dataset/
├── source/    # Satellite images
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── masks/     # Corresponding segmentation masks
    ├── mask_001.png
    ├── mask_002.png
    └── ...
```

#### 2. Data Augmentation
```python
from albumentations import Compose, HorizontalFlip, VerticalFlip

transform = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5)
])
```

#### 3. Dataset Class
- Custom `ImageSegmentationDataset` class
- Loads images and masks from disk
- Applies transformations
- Encodes inputs using `SegformerFeatureExtractor`
- Resizes to 256x256 pixels

#### 4. Data Splitting
- **Training**: 70%
- **Validation**: 15%
- **Testing**: 15%
- Random seed: 42 (for reproducibility)

#### 5. Model Configuration
```python
id2label = {0: "background", 1: "building"}
label2id = {"background": 0, "building": 1}

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b4",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    reshape_last_stage=True,
    image_size=256
)
```

#### 6. Training Parameters
- **Optimizer**: AdamW (lr=0.00006) or SGD (lr=0.01, momentum=0.9)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 4
- **Epochs**: 10-20
- **Mixed Precision**: AMP (Automatic Mixed Precision) enabled
- **Gradient Accumulation**: 8 steps (for larger effective batch sizes)

#### 7. Evaluation Metrics
- **Pixel-wise Accuracy**: Percentage of correctly classified pixels
- **Intersection over Union (IoU)**: Jaccard index for segmentation quality
- **Loss**: Cross-entropy loss value

#### 8. Advanced Techniques
- **Encoder Freezing**: Option to freeze early encoder blocks
- **Progressive Unfreezing**: Fine-tune later layers while keeping early layers frozen
- **Learning Rate Scheduling**: Adjust learning rate during training

#### 9. Model Saving & Loading
```python
# Save model
torch.save(model.state_dict(), 'segformer_model.pth')

# Load model
model.load_state_dict(torch.load('segformer_model.pth'))
```

### Training Best Practices
1. Start with frozen encoder blocks, then unfreeze gradually
2. Monitor validation metrics to prevent overfitting
3. Use data augmentation to improve generalization
4. Train for 10-20 epochs with early stopping
5. Save checkpoints at regular intervals

---

## API Endpoints

### 1. Home Page
```
GET /
```
Returns the main HTML interface.

### 2. Process GeoTIFF
```
POST /process
Content-Type: multipart/form-data

Parameters:
- file: GeoTIFF file (.tif or .tiff)

Response:
{
    "success": true,
    "result_file": "filename_result.geojson",
    "mask_file": "filename_mask.png",
    "input_file": "filename_input.png",
    "total_area": "24500.0 m²",
    "total_area_m2": 24500.0,
    "total_generation": 1234567.89,
    "total_power": "441.00 kW",
    "panel_info": {
        "type": "Monocrystalline PERC",
        "efficiency": 18.0,
        "lifespan": 25
    },
    "geojson": { ... }
}
```

### 3. Process ROI
```
POST /process_roi
Content-Type: application/json

Body:
{
    "north": 12.9716,
    "south": 12.9616,
    "east": 77.6016,
    "west": 77.5916
}

Response: Same as /process endpoint
```

### 4. Get Results
```
GET /results/<filename>
```
Returns the requested file (image or GeoJSON).

---

## Project Structure

```
Udayachal_Project_code/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── segformer_main.ipynb           # Model training notebook (Jupyter)
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── model/
│   ├── segformer_model.pth        # Trained model weights (not included)
│   └── predict_utils.py            # Prediction utilities (placeholder)
│
├── data/
│   └── irradiance_data.csv        # Solar irradiance data (NASA SSE)
│
├── templates/
│   └── index.html                  # Main web interface
│
├── uploads/                        # Uploaded GeoTIFF files (auto-created)
│
├── results/                        # Processed results (auto-created)
│   ├── *_input.png                # Input image visualizations
│   ├── *_mask.png                 # Prediction mask visualizations
│   └── *_result.geojson           # GeoJSON results
│
└── static/
    └── predictions/                # Static prediction images
```

---

## Data Pipeline

### GeoTIFF Processing Pipeline

```
1. File Upload
   ↓
2. Validation (format, size)
   ↓
3. Open with Rasterio
   ↓
4. CRS Detection & Resolution Calculation
   ↓
5. Resample to 0.6m resolution
   ↓
6. Convert to RGB (if necessary)
   ↓
7. Normalize to uint8 [0-255]
   ↓
8. Save input visualization
   ↓
9. Pass through SegFormer model
   ↓
10. Generate prediction mask
   ↓
11. Convert mask to GeoJSON with georeferencing
   ↓
12. Calculate area for each feature (UTM projection)
   ↓
13. Lookup irradiance data (bilinear interpolation)
   ↓
14. Calculate solar potential metrics
   ↓
15. Return results as JSON
```

### ROI Processing Pipeline

```
1. User draws rectangle on map
   ↓
2. Extract bounding box coordinates
   ↓
3. Validate against Karnataka boundaries
   ↓
4. Calculate optimal zoom level (19)
   ↓
5. Generate tile list using Mercantile
   ↓
6. Download Google Satellite tiles
   ↓
7. Create mosaic from tiles
   ↓
8. Save input visualization
   ↓
9. Process in patches (512x512, 64px overlap)
   ↓
10. Stitch predictions together
    ↓
11. Create georeferenced transform
    ↓
12. Convert to GeoJSON
    ↓
13. Calculate solar potential
    ↓
14. Return results
```

---

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t udayachal .
docker run -p 5000:5000 udayachal
```

### Cloud Deployment Options

#### Heroku
1. Create `Procfile`:
   ```
   web: gunicorn app:app
   ```
2. Deploy:
   ```bash
   heroku create udayachal-app
   git push heroku main
   ```

#### AWS EC2
1. Launch Ubuntu EC2 instance
2. Install dependencies
3. Clone repository
4. Run with systemd service

#### Google Cloud Run
1. Containerize with Docker
2. Push to Google Container Registry
3. Deploy to Cloud Run

### Important Deployment Notes

⚠️ **GEE API Integration**: 
For production deployment with Google Earth Engine map functionality, you need to:
1. Set up a Google Cloud Platform account
2. Enable the Earth Engine API
3. Generate API credentials
4. Add authentication to `app.py`:
   ```python
   import ee
   ee.Authenticate()
   ee.Initialize()
   ```

⚠️ **Model File**: 
Ensure `model/segformer_model.pth` is included in your deployment package.

⚠️ **Environment Variables**: 
For production, move sensitive configurations to environment variables:
```python
import os
MAPBOX_TOKEN = os.environ.get('MAPBOX_TOKEN', 'default_token')
```

---

## Performance Metrics

### Model Performance (Typical)
Based on training results from `segformer_main.ipynb`:

| Metric | Training Set | Validation Set | Test Set |
|--------|--------------|----------------|----------|
| **Pixel Accuracy** | 0.92-0.95 | 0.88-0.92 | 0.87-0.91 |
| **IoU (Building)** | 0.85-0.89 | 0.80-0.85 | 0.78-0.83 |
| **Loss** | 0.15-0.20 | 0.20-0.28 | 0.22-0.30 |

### Inference Performance
- **Small Images (< 1000x1000)**: 2-5 seconds
- **Medium Images (1000-5000 pixels)**: 10-20 seconds
- **Large Images (> 5000 pixels)**: 30-60 seconds (patch-based processing)
- **ROI Processing**: 15-30 seconds (depends on tile count)

### Memory Usage
- **Model Loading**: ~500 MB (MiT-B4)
- **Inference**: 1-2 GB (depending on image size)
- **Peak Memory**: 3-4 GB (large GeoTIFF processing)

### Optimization Tips
1. **Use GPU**: CUDA acceleration provides 5-10x speedup
2. **Adjust Patch Size**: Larger patches = faster processing, more memory
3. **Batch Processing**: Process multiple images in production
4. **Caching**: Cache irradiance lookups for repeated locations
5. **Model Quantization**: Reduce model size for deployment (INT8 quantization)

---

## Troubleshooting

### Common Issues

#### 1. Model Loading Error
**Error**: `FileNotFoundError: model/segformer_model.pth`
**Solution**: Ensure the trained model file is in the `model/` directory.

#### 2. CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`
**Solution**: 
- Reduce `PATCH_SIZE` in `app.py`
- Use CPU: `device = torch.device('cpu')`
- Close other GPU-intensive applications

#### 3. Invalid File Type
**Error**: "Invalid file type"
**Solution**: Ensure you're uploading `.tif` or `.tiff` files only.

#### 4. Karnataka Boundary Error
**Error**: "Selected region is outside Karnataka"
**Solution**: The ROI tool only works within Karnataka boundaries (11.5°N to 18.4°N, 74.0°E to 78.6°E).

#### 5. Irradiance Data Not Found
**Error**: "No irradiance data points found"
**Solution**: Verify `data/irradiance_data.csv` exists and contains valid data for your location.

#### 6. Import Errors
**Error**: `ModuleNotFoundError`
**Solution**: 
```bash
pip install --upgrade -r requirements.txt
```

#### 7. Rasterio Installation Failure
**Error**: Compilation errors during rasterio install
**Solution**: 
```bash
# Windows
conda install -c conda-forge rasterio

# Linux
sudo apt-get install gdal-bin libgdal-dev
pip install rasterio
```

### Debug Mode
Enable detailed logging by checking the console output when running `app.py`. The application uses Python's logging module with INFO level by default.

### Performance Debugging
Add timing measurements:
```python
import time
start_time = time.time()
# ... your code ...
print(f"Processing time: {time.time() - start_time:.2f} seconds")
```

---

## Future Enhancements

### Short-term
- [ ] Support for multi-class segmentation (residential, commercial, industrial rooftops)
- [ ] Real-time processing progress bar
- [ ] Download results as PDF report
- [ ] Historical irradiance data comparison
- [ ] Export results to Shapefile format

### Medium-term
- [ ] Integration with weather forecasting APIs
- [ ] Economic analysis (ROI, payback period)
- [ ] Support for additional Indian states
- [ ] Mobile-responsive interface improvements
- [ ] User authentication and history tracking

### Long-term
- [ ] 3D rooftop modeling with shadow analysis
- [ ] Time-series solar generation predictions
- [ ] Integration with smart grid systems
- [ ] Automated permit generation
- [ ] Multi-language support (Kannada, Hindi, English)

---

## Contributing

We welcome contributions to Udayachal! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Your Changes**: `git commit -m 'Add amazing feature'`
4. **Push to the Branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 coding standards
- Add docstrings to all functions
- Include type hints where possible
- Write unit tests for new features
- Update documentation accordingly

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

**Note**: The model training code and trained weights are maintained separately and may have different licensing terms. Contact the maintainers for commercial use inquiries.

---

## Acknowledgments

### Data Sources
- **NASA SSE (Surface meteorology and Solar Energy)**: Solar irradiance data
- **Google Maps Platform**: Satellite imagery tiles
- **OpenStreetMap**: Geographic reference data

### Libraries & Frameworks
- **Hugging Face Transformers**: SegFormer model implementation
- **Leaflet**: Interactive web mapping
- **Flask**: Web application framework
- **PyTorch**: Deep learning framework

### Inspiration
This project was inspired by the growing need for renewable energy solutions in India and the potential of AI to accelerate solar adoption.

### Team
- **Developer**: [Your Name/Team]
- **Institution**: Udayachal Project
- **Region**: Karnataka, India

---

## Contact & Support

- **GitHub Repository**: https://github.com/Viratcode/Udayachal_Project_code
- **Issues**: Report bugs and request features via GitHub Issues
- **Email**: [Your Contact Email]
- **Documentation**: This README serves as the primary documentation

---

## Project Statistics

- **Lines of Code**: ~2,500+ (app.py + segformer_main.ipynb)
- **Model Parameters**: ~60M (SegFormer MiT-B4)
- **Supported File Formats**: GeoTIFF, PNG, GeoJSON
- **Processing Resolution**: 0.6 meters per pixel
- **Coverage Area**: Karnataka, India (202,000+ km²)

---

**Made with ❤️ for sustainable energy future**

*Udayachal - Illuminating Karnataka's Solar Potential*
