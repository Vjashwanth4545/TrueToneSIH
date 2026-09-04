import os
import tempfile
import logging

import torch
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Configure logging to catch detailed errors in terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================
# CREATE FASTAPI APP
# =====================================

app = FastAPI(title="Deepfake Audio Detection API")

# =====================================
# ALLOW YOUR HTML FRONTEND
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# DEVICE SETUP
# =====================================

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")

# =====================================
# LOAD MODEL & FEATURE EXTRACTOR
# =====================================

MODEL_PATH = "bestmodel.pth"

# 1. Initialize feature extractor to properly normalize raw audio inputs
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# 2. Initialize model architecture
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=2
)

# 3. Load checkpoint
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Handle checkpoints saved as pure state_dict vs dictionary wrapper
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Custom model weights loaded successfully")
else:
    print(f"⚠️ Warning: Checkpoint file '{MODEL_PATH}' not found!")

model.to(device)
model.eval()

# =====================================
# LABELS
# =====================================

labels = {
    0: "FAKE",
    1: "REAL"
}

# =====================================
# HOME API
# =====================================

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Deepfake Audio Detection API is running"
    }

# =====================================
# PREDICT API
# =====================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
    filename = file.filename.lower()

    if not filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {allowed_extensions}"
        )

    suffix = os.path.splitext(file.filename)[1]

    # Use standard context writing to ensure data flushes to disk properly
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        content = await file.read()
        temp_file.write(content)
        temp_file.flush()  # Force OS to write remaining bytes to disk

    try:
        # Load audio at 16kHz
        audio, sr = librosa.load(temp_path, sr=16000, mono=True)

        # Standardize to 4 seconds (64,000 samples)
        MAX_LENGTH = 16000 * 4

        if len(audio) > MAX_LENGTH:
            audio = audio[:MAX_LENGTH]
        elif len(audio) < MAX_LENGTH:
            padding = MAX_LENGTH - len(audio)
            audio = torch.nn.functional.pad(
                torch.tensor(audio),
                (0, padding)
            ).numpy()

        # Process input using Hugging Face's Feature Extractor (normalizes values)
        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_values=input_values)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        pred_idx = prediction.item()
        conf_val = confidence.item()
        fake_prob = probabilities[0][0].item()
        real_prob = probabilities[0][1].item()

        return {
            "success": True,
            "filename": file.filename,
            "prediction": labels.get(pred_idx, "UNKNOWN"),
            "confidence": round(conf_val * 100, 2),
            "fake_probability": round(fake_prob * 100, 2),
            "real_probability": round(real_prob * 100, 2)
        }

    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")