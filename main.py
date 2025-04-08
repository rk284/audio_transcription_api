# Audio-to-Text Transcription API (Complete Project)
# Features: Speaker Diarization, Multilingual Support, Punctuation, Translation
# Tools Used: pyannote-audio, Facebook wav2vec2, DeepMultilingualPunctuation, Facebook M2M100

# ========================= SETUP =========================
# Create a Python virtual environment and install dependencies

# 1. Install these in requirements.txt:
# torch==1.13.1
# torchaudio==0.13.1
# transformers==4.26.1
# pyannote.audio==2.1.1
# deepmultilingualpunctuation
# librosa
# pydub
# fastapi
# uvicorn
# python-multipart
# aiofiles

# In terminal (VS Code):
# python -m venv venv
# source venv/bin/activate (or venv\Scripts\activate on Windows)
# pip install -r requirements.txt

# ========================= MODEL LOADING =========================

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from pathlib import Path
import torch
import librosa
import os
import tempfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deepmultilingualpunctuation import PunctuationModel
from pyannote.audio import Pipeline

app = FastAPI()

# Load Wav2Vec2 model for English transcription
asr_model_name = "facebook/wav2vec2-base-960h"
asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained(asr_model_name)
asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)

# Load punctuation model
punctuation_model = PunctuationModel()

# Load translation model
translation_model_name = "facebook/m2m100_418M"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

# Load speaker diarization pipeline from pyannote (requires Hugging Face auth token)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="<your_huggingface_token>")

# ========================= HELPER FUNCTIONS =========================

def convert_audio(file_path):
    # Convert audio to 16kHz mono wav (required by models)
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_path = tempfile.mktemp(suffix=".wav")
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(wav_path):
    speech, rate = librosa.load(wav_path, sr=16000)
    input_values = asr_tokenizer(speech, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_tokenizer.batch_decode(predicted_ids)[0]
    return transcription.lower()

def punctuate(text):
    return punctuation_model.restore_punctuation(text)

def translate(text, target_lang="fr"):
    translation_tokenizer.src_lang = "en"
    encoded = translation_tokenizer(text, return_tensors="pt")
    generated = translation_model.generate(**encoded, forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang))
    return translation_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def diarize_audio(wav_path):
    diarization = pipeline(wav_path)
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return results 


# ========================= API ENDPOINT =========================

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    target_language: str = Form("fr")
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name

        # Convert and diarize audio
        wav_path = convert_audio(audio_path)
        diarization = diarize_audio(wav_path)

        # Transcribe audio
        transcription = transcribe_audio(wav_path)

        # Add punctuation
        punctuated = punctuate(transcription)

        # Translate
        translated = translate(punctuated, target_language)

        return JSONResponse(content={
            "speaker_diarization": diarization,
            "original_transcription": punctuated,
            "translated_transcription": translated
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ========================= RUN LOCALLY =========================
# Run this command in terminal:
# uvicorn main:app --reload


# ========================= SAMPLE REQUEST =========================
# Use curl or Postman to test:
# curl -X POST "http://localhost:8000/transcribe" -F "file=@your_audio.mp3" -F "target_language=fr"
