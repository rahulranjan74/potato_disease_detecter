from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1")

CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image, 0)
    load_options = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])


    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }







if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
