import os
import torch

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # We do not need training data paths for the inference API
    DATA_DIR = None 
    CAPTIONS_FILE = None
    
    # Model saving/loading directory
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Device: Force CPU if CUDA is not available (Hugging Face Free Tier is CPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters (kept for reference, mostly unused in inference)
    BATCH_SIZE = 1
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    NUM_WORKERS = 2
    
    # Model Config
    # Change this to "blip" or "vit_gpt2" for your deployment to ensure no custom weights are needed
    MODEL_TYPE = "blip"        
    ENCODER_MODEL = "resnet50" 
    DECODER_MODEL = "gpt2"     
    EMBED_DIM = 768            
    MAX_SEQ_LEN = 40
    
    # Image Config
    IMAGE_SIZE = (224, 224)

config = Config()