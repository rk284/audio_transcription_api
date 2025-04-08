# Audio-to-Text Transcription API

An AI-driven transcription API that converts audio files to accurately structured text. The project handles:
- **Speaker Diarization** using [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- **Speech-to-text transcription** with [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- **Punctuation restoration** with [deepmultilingualpunctuation](https://github.com/MaxWolf117/DeepMultilingualPunctuation)
- **Multilingual translation** using [Facebook’s M2M100 model](https://huggingface.co/facebook/m2m100_418M)

## Features

- **Speaker Diarization:** Detects and labels individual speakers in the audio.
- **Accurate Transcriptions:** Uses state-of-the-art ASR models.
- **Automatic Punctuation:** Restores punctuation for clarity.
- **Multilingual Support:** Translates the transcription to a target language as specified by the user.
- **FastAPI Integration:** Provides a RESTful API interface for transcription.

## Demo

Send a POST request to `/transcribe` with an audio file and target language:
- **Endpoint:** `http://localhost:8000/transcribe`
- **Method:** POST
- **Form Data:**
  - `file`: Your audio file (e.g., `sample.mp3`)
  - `target_language`: Target language code (e.g., `fr` for French)

Example cURL command:

curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@sample.mp3" \
  -F "target_language=fr"

# Setup Instructions
## Prerequisites
 - Python 3.7 or higher
 - Git installed

## Steps
- Clone the Repository: git clone https://github.com/<your_username>/audio_transcription_api.git
- Create a Virtual Environment and Activate It:
 

## Install Dependencies:
pip install -r requirements.txt


## Add your Hugging Face token:
<your_huggingface_token> = your_actual_hf_token

# Run the API Server:
uvicorn main:app --reload

# Test the API:
Open http://localhost:8000/docs in your browser for the interactive documentation, or use cURL/Postman as described above.
