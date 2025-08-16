MobileNetV2 Quantization & Deployment
```
├── mobilenetv2_finetuning.ipynb # Training + fine-tuning MobileNetV2 on CIFAR-10
├── Quantized_mobilenetv2.ipynb # Post-training quantization and TFLite conversion
├── mobilenetv2_cifar10.tflite # Standard TFLite model
├── mobilenetv2_cifar10_postquantized.tflite # Quantized TFLite model
├── main.py # FastAPI backend for inference
├── static/index.html # Simple frontend to upload images & test
├── requirements.txt # Dependencies

###  Clone the repo
```bash
git clone https://github.com/rohitdasari1/mobilenetv2-quantization-deployment.git
cd mobilenetv2-quantization-deployment

python3 -m venv venv
source venv/bin/activate   # On Ubuntu / Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt

uvicorn main:app --reload

This will start the backend at:

http://127.0.0.1:8000

Open the frontend

Go to:

http://127.0.0.1:8000/static/index.html

 Dependencies

Python 3.8+

TensorFlow / TensorFlow Lite Runtime

FastAPI

Uvicorn

Pillow

NumPy

(Already included in requirements.txt)

Results
| Model Variant                    | Size Reduction  | Accuracy on CIFAR-10 |
| -------------------------------- | --------------- | -------------------- |
| **Fine-tuned MobileNetV2**       | 100% (baseline) | **91.00%**           |
| **Quantized MobileNetV2 (INT8)** | \~4× smaller    | **87.18%**           |


Quantization led to a huge reduction in model size while keeping accuracy drop minimal (-3.82%).
This makes the model lightweight and deployment-ready for mobile and edge devices.
