import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.strip()
    return text

def tokenize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])
