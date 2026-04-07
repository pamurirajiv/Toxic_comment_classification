import os
import pandas as pd
import tensorflow as tf
from preprocessing import clean_text
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    GlobalMaxPooling1D
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, AUC


LOCAL_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(LOCAL_DIR, "train (2).csv")
MODEL_PATH = os.path.join(LOCAL_DIR, "model.h5")
VECTORIZER_PATH = os.path.join(LOCAL_DIR, "vectorizer.keras")


df = pd.read_csv(DATA_PATH)

if "id" in df.columns:
    df = df.drop(columns=["id"])


df["comment_text"] = df["comment_text"].apply(clean_text)


label_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

x_train = train["comment_text"]
y_train = train[label_columns].values

x_test = test["comment_text"]
y_test = test[label_columns].values


MAX_FEATURES = 200000
SEQ_LENGTH = 200

vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=SEQ_LENGTH,
    output_mode="int"
)

vectorizer.adapt(x_train.values)


vectorizer_model = tf.keras.Sequential([vectorizer])
vectorizer_model.save(VECTORIZER_PATH)


x_train_vectorized = vectorizer(x_train.values)
x_test_vectorized = vectorizer(x_test.values)


model = Sequential()

model.add(
    Embedding(
        input_dim=MAX_FEATURES,
        output_dim=128
    )
)

model.add(
    Bidirectional(
        LSTM(128, return_sequences=True)
    )
)

model.add(GlobalMaxPooling1D())

model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))

model.add(Dense(6, activation="sigmoid"))


model.compile(
    loss='binary_crossentropy',
    optimizer="adam",
    metrics=[
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc")
    ]
)


model.summary()


history = model.fit(
    x_train_vectorized,
    y_train,
    epochs=8,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


loss, precision, recall, auc = model.evaluate(
    x_test_vectorized,
    y_test
)

print(f"Loss: {loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")

model.save(MODEL_PATH)