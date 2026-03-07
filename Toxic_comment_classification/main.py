import os
import re
import logging
from preprocessing import clean_text
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LOCAL_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(LOCAL_DIR, "model.h5")

VECTORIZER_PATH = os.path.join(LOCAL_DIR, "vectorizer.keras")

model = None
vectorizer = None

class CommentRequest(BaseModel):
    comment: str

@app.on_event("startup")
def load_artifacts():

    global model, vectorizer

    try:

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        vectorizer = tf.keras.models.load_model(VECTORIZER_PATH)

        logger.info("Model and vectorizer loaded successfully")

    except Exception as e:

        logger.error("Failed to load artifacts", exc_info=True)

        raise e

@app.post("/predict")
def predict(data: CommentRequest):

    comment = clean_text(data.comment)

    x_test = vectorizer(tf.constant([comment]))

    predictions = model.predict(x_test, verbose=0)[0]

    labels = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    ]

    result = {label: float(score) for label, score in zip(labels, predictions)}

    return {
        "comment": comment,
        "prediction": result
    }


@app.get("/health")
def health_check():

    if model is None:
        return {"status": "model not loaded"}

    return {"status": "ok"}