import re


def clean_text(text):

    text = str(text)

    text = re.sub(r"[^\x00-\x7F]+", "", text)

   
    text = text.lower()

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


