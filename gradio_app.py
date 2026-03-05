import gradio as gr
import requests


API_URL = "http://127.0.0.1:8000/predict"


def predict_comment(comment):

    payload = {
        "comment": comment
    }

    response = requests.post(API_URL, json=payload)

    result = response.json()["prediction"]

    return result


interface = gr.Interface(
    fn=predict_comment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter a comment..."
    ),
    outputs=gr.JSON(),
    title="Toxic Comment Detection",
    description="Detect toxicity in online comments using a BiLSTM model"
)


interface.launch()