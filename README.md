#  MobileNetV2 Quantization & Deployment  

---

##  Processing Flow
1. **Train & Fine-tune** MobileNetV2 on CIFAR-10 (`mobilenetv2_finetuning.ipynb`)  
2. **Convert to TFLite** (`Quantized_mobilenetv2.ipynb`)  
   - Standard TFLite model  
   - Quantized TFLite model (INT8)  
3. **Deploy with FastAPI** (`main.py`)  
4. **Test via Frontend** (`static/index.html`)  

---

##  Project Structure
.
├── mobilenetv2_finetuning.ipynb # Training + fine-tuning on CIFAR-10
├── Quantized_mobilenetv2.ipynb # Post-training quantization + TFLite
├── mobilenetv2_cifar10.tflite # Standard TFLite model
├── mobilenetv2_cifar10_postquantized.tflite # Quantized TFLite model
├── main.py # FastAPI backend for inference
├── static/index.html # Simple frontend for testing
├── requirements.txt # Dependencies


---

##  Installation
```bash
git clone https://github.com/rohitdasari1/mobilenetv2-quantization-deployment.git
cd mobilenetv2-quantization-deployment

python3 -m venv venv
source venv/bin/activate   # On Ubuntu / Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt

Running the Backend
uvicorn main:app --reload
Backend runs at: http://127.0.0.1:8000

Access the Frontend
open in browser:  http://127.0.0.1:8000/static/index.html


```

Results:
| Model Variant                    | Size Reduction | Accuracy on CIFAR-10 |
| -------------------------------- | -------------- | -------------------- |
| **Fine-tuned MobileNetV2**       | Baseline       | **91.00%**           |
| **Quantized MobileNetV2 (INT8)** | \~4× smaller   | **87.18%**           |

**Quantization reduced the model size drastically with only -3.82% accuracy drop.
The final model is lightweight & optimized for mobile/edge deployment.**
