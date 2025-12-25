import io
import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import GPT2Tokenizer

# Adjust imports based on your folder structure
from config.config import config
from src.models.model import get_model
from src.preprocessing.transforms import get_transforms

app = FastAPI(title="Object Captioning LLM API")

# --- CORS CONFIGURATION (Crucial for Vercel) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
print(f"Loading model: {config.MODEL_TYPE} on {config.DEVICE}...")
device = config.DEVICE
model = get_model(config).to(device)

# Legacy support for ResNetGPT2
if config.MODEL_TYPE == "resnet_gpt2":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Use a relative path or ensure this file is uploaded to the Docker container
    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model_llm.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained custom model.")
    else:
        print("Warning: No trained model found for ResNetGPT2. Using random weights.")
else:
    tokenizer = None 

model.eval()
transform = get_transforms(train=False)

@app.get("/")
def home():
    return {"message": "Image Captioning API is running. Send POST requests to /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read Image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Generate Caption
        if config.MODEL_TYPE == "resnet_gpt2":
            img_tensor = transform(image).to(device)
            # Ensure generate_caption handles the tensor/tokenizer correctly
            caption = model.generate_caption(img_tensor, tokenizer)
        else:
            # SOTA models (BLIP/ViT) take the PIL image directly
            caption = model.generate_caption(image)
        
        return {
            "caption": caption
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))