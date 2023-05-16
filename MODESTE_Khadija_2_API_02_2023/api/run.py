from fastapi import FastAPI
from fastapi import UploadFile, File
from preprocessing import *
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO


app = FastAPI()


@app.get("/")
async def hello_wold():
    return "Hello world"

@app.post("/predict")
async def predict_mask(img: UploadFile = File(...)):
    original_image = Image.open(img.file)
    original_image = original_image.convert('RGB')
    original_image = original_image.resize((256, 128))
    mask = predict_img_to_color_mask(original_image)
    filtered_image = BytesIO()
    mask.save(filtered_image, "png")
    filtered_image.seek(0)

    return StreamingResponse(filtered_image, media_type="image/png")
