from fastapi import FastAPI, UploadFile
import uvicorn
import tempfile
from model.predict_utils import process_image

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    result_geojson = process_image(temp_path)
    return result_geojson  # Return GeoJSON directly

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
