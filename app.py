from fastapi import FastAPI, UploadFile
from model import SimpleUNet
from utils import mask_to_geojson
import shutil

app = FastAPI()
model = SimpleUNet()

@app.post("/analyze")
async def analyze(file: UploadFile):
    filepath = f"/tmp/{file.filename}"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    mask = model.predict(filepath)
    geojson = mask_to_geojson(mask)
    return geojson

