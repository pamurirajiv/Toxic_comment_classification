import tensorflow as tf
import os

LOCAL_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(LOCAL_DIR, "model.h5")
VECTORIZER_PATH = os.path.join(LOCAL_DIR, "vectorizer.keras")


model = tf.keras.models.load_model(MODEL_PATH, compile=False)

vectorizer = tf.keras.models.load_model(VECTORIZER_PATH)

comments = [
    "I hate you",
    "You are amazing",
    "I will kill you",
    "I do not hate you"
]


x_test = vectorizer(tf.constant([comments]))

predictions = model.predict(x_test)

labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

for comment, pred in zip(comments, predictions):
    print("\nComment:", comment)
    for label, score in zip(labels, pred):
        print(f"{label}: {score:.3f}")