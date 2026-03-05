
import re

def clean_text(text):
    text = str(text)

    # remove non ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # convert to lowercase
    text = text.lower()

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
