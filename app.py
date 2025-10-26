from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import re
import emoji
import spacy
import json
import logging
import os
from bs4 import BeautifulSoup
from collections import defaultdict
from transformers import pipeline
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
EMO_SERVICE_API_KEY = os.getenv("EMO_SERVICE_API_KEY")

# --- FastAPI app ---
app = FastAPI(title="Emotion Detection API", version="1.0")

# --- Request body model ---
class EmotionRequest(BaseModel):
    apikey: str
    text: str


# --- Emotion detection function ---
def get_emotion(text: str) -> dict:
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Load Hugging Face model
    try:
        emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1,
            trust_remote_code=False
        )
    except Exception as e:
        return {"error": f"Failed to load emotion model: {str(e)}"}

    EMOTION_UI_MAP = {
        "joy": "😄 Joy",
        "sadness": "😢 Sadness",
        "anger": "😠 Anger",
        "fear": "😨 Fear",
        "disgust": "🤢 Disgust",
        "surprise": "😲 Surprise",
        "neutral": "😐 Neutral"
    }

    def clean_text(text: str) -> str:
        text = BeautifulSoup(text, "html.parser").get_text()
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_sentences(text: str, min_length: int = 4):
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= min_length]

    def detect_emotion(sentence: str):
        try:
            result = emotion_model(sentence)[0]
            return {res["label"]: res["score"] for res in result}
        except Exception as e:
            logger.error(f"Prediction failed for sentence: {sentence} | Error: {e}")
            return {}

    if not isinstance(text, str) or len(text.strip().split()) < 3:
        return {"error": "Input text is too short for a meaningful analysis."}

    cleaned_text = clean_text(text)
    sentences = split_sentences(cleaned_text)
    if not sentences:
        return {"error": "No valid sentences were found after cleaning."}

    emotion_totals = defaultdict(float)
    for sent in sentences:
        scores = detect_emotion(sent)
        for emo, score in scores.items():
            emotion_totals[emo] += score

    total_sentences = len(sentences)
    threshold = 0.01
    top_k = 7

    avg_scores = {
        emo: round(score / total_sentences, 4)
        for emo, score in emotion_totals.items()
    }

    filtered_scores = {k: v for k, v in avg_scores.items() if v >= threshold}
    top_scores = dict(sorted(filtered_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k])

    result = {
        EMOTION_UI_MAP.get(k, k.capitalize()): v
        for k, v in top_scores.items()
    }

    if not result:
        result = {"😐 Neutral": 1.0}

    return result


# --- API routes ---
@app.get("/")
def home():
    return {"message": "Emotion Detection API is running 🚀"}


@app.post("/get_emotion")
async def emotion_api(data: EmotionRequest):
    if data.apikey != EMO_SERVICE_API_KEY:
        return JSONResponse(status_code=403, content={"error": "Invalid API key."})

    result = get_emotion(data.text)
    return JSONResponse(content=result)
