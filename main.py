import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# CIFAR-10 labels
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# Load quantized TFLite model
interpreter = tf.lite.Interpreter(model_path="mobilenetv2_cifar10_postquantized.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Quantization params
input_scale, input_zero_point = input_details[0]['quantization']

# FastAPI setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS (for frontend JS fetch)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image: Image.Image):
    # Resize to 224x224 (MobileNetV2 input size)
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # Quantize if model expects int
    if input_details[0]['dtype'] in [np.uint8, np.int8]:
        img = img / input_scale + input_zero_point
        img = np.clip(img, 0, 255).astype(input_details[0]['dtype'])

    return np.expand_dims(img, axis=0)

@app.get("/", response_class=HTMLResponse)
async def homepage():
    with open("static/index.html") as f:
        return f.read()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_class = CLASS_NAMES[np.argmax(output)]
    return {"prediction": pred_class}
