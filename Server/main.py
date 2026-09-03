import os
import tempfile

import torch
import librosa

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2ForSequenceClassification


# =====================================
# CREATE FASTAPI APP
# =====================================

app = FastAPI(title="Deepfake Audio Detection API")


# =====================================
# ALLOW YOUR HTML FRONTEND
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# DEVICE
# =====================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# =====================================
# LOAD MODEL
# =====================================

MODEL_PATH = "bestmodel.pth"

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

print("✅ Model loaded successfully")


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

    # Allowed formats
    allowed_extensions = (
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg"
    )

    filename = file.filename.lower()

    if not filename.endswith(allowed_extensions):

        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format"
        )


    # Create temporary file
    suffix = os.path.splitext(file.filename)[1]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_path = temp_file.name


    try:

        # Save uploaded audio
        content = await file.read()

        temp_file.write(content)

        temp_file.close()


        # =====================================
        # LOAD AUDIO
        # =====================================

        audio, sr = librosa.load(
            temp_path,
            sr=16000,
            mono=True
        )


        # =====================================
        # STANDARDIZE TO 4 SECONDS
        # =====================================

        MAX_LENGTH = 16000 * 4

        if len(audio) > MAX_LENGTH:

            audio = audio[:MAX_LENGTH]

        elif len(audio) < MAX_LENGTH:

            padding = MAX_LENGTH - len(audio)

            audio = torch.nn.functional.pad(
                torch.tensor(audio),
                (0, padding)
            ).numpy()


        # =====================================
        # MODEL INPUT
        # =====================================

        audio_tensor = torch.tensor(
            audio,
            dtype=torch.float32
        ).unsqueeze(0).to(device)


        # =====================================
        # PREDICTION
        # =====================================

        with torch.no_grad():

            outputs = model(
                input_values=audio_tensor
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )


        # =====================================
        # RESULTS
        # =====================================

        prediction = prediction.item()

        confidence = confidence.item()

        fake_probability = probabilities[0][0].item()

        real_probability = probabilities[0][1].item()


        return {

            "success": True,

            "filename": file.filename,

            "prediction": labels[prediction],

            "confidence": round(
                confidence * 100,
                2
            ),

            "fake_probability": round(
                fake_probability * 100,
                2
            ),

            "real_probability": round(
                real_probability * 100,
                2
            )
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        if os.path.exists(temp_path):

            os.remove(temp_path)