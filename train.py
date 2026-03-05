import os
from preprocessing import clean_text
import pandas as pd
import tensorflow as tf
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
from tensorflow.keras.metrics import Precision, Recall


LOCAL_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(LOCAL_DIR, "train (2).csv")

MODEL_PATH = os.path.join(LOCAL_DIR, "model.h5")

VECTORIZER_PATH = os.path.join(LOCAL_DIR, "vectorizer.keras")


df = pd.read_csv(DATA_PATH)

if "id" in df.columns:
    df = df.drop(columns=["id"])

df["comment_text"] = df["comment_text"].apply(clean_text)

df = df.sample(frac=0.4, random_state=42).reset_index(drop=True)


train, test = train_test_split(df, test_size=0.2, random_state=42)

x_train = train["comment_text"]

y_train = train[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values


MAX_FEATURES = 200000
SEQ_LENGTH = 200


vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=SEQ_LENGTH,
    output_mode="int"
)


vectorizer.adapt(x_train.values)

vectorizer(["sample sentence"])

vectorizer_model = tf.keras.Sequential([vectorizer])
vectorizer_model.save(VECTORIZER_PATH)

x_train_vectorized = vectorizer(x_train.values)


model = Sequential()

model.add(
    Embedding(
        input_dim=MAX_FEATURES,
        output_dim=128,
        input_length=SEQ_LENGTH
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


# build model
model.build(input_shape=(None, SEQ_LENGTH))

model.summary()

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[Precision(), Recall()]
)

history = model.fit(
    x_train_vectorized,
    y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


x_test = vectorizer(test["comment_text"].values)

y_test = test[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values



print(model.evaluate(x_test, y_test))


model.save(MODEL_PATH)
