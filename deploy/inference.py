import os
import re
import json
import pickle

import torch

from model import BiLSTMClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

OPTION_LABELS = ["A", "B", "C", "D", "E"]

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------
# Preprocessing — identical to training
# --------------------------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_prompt(prompt):          

    patterns = [
        r"pick the best possible answer:",
        r"choose the correct answer:",
        r"determine the correct option:",
        r"select the most accurate option:",
        r"what is the correct answer\?",
        r"among the listed options\.?",
        r"from the following choices\.?",
        r"based on the given context\.?"
    ]

    for p in patterns:
        prompt = re.sub(p, "", prompt)

    prompt = re.sub(r"\s+", " ", prompt)

    return prompt.strip()


# --------------------------------------------------------------------
# Load artifacts once at import time
# --------------------------------------------------------------------

def _load_config():
    with open(os.path.join(MODELS_DIR, "config.json"), "r") as f:
        return json.load(f)


def _load_tokenizer():
    with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb") as f:
        return pickle.load(f)


def _load_model(config):
    model = BiLSTMClassifier(
        vocab_size=config["VOCAB_SIZE"],
        embed_dim=config["EMBED_DIM"],
        hidden_dim=config["HIDDEN_DIM"],
        dropout=config["DROPOUT"],
    )
    state_dict = torch.load(
        os.path.join(MODELS_DIR, "model.pth"),
        map_location=_device,
    )
    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()
    return model


_config = _load_config()
_tokenizer = _load_tokenizer()
_model = _load_model(_config)

MAX_LENGTH = _config["MAX_LENGTH"]


# --------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------

def predict(prompt: str, options: dict):
    """
    prompt: raw question text
    options: dict like {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}
             (any subset/order of OPTION_LABELS is fine)

    Returns: list of (option_label, probability) sorted by probability
             descending, e.g. [("B", 0.973), ("D", 0.921), ("A", 0.865), ...]
    """
    labels_present = [lab for lab in OPTION_LABELS if lab in options]
    if not labels_present:
        raise ValueError("No valid options (A-E) provided.")

    # Same preprocessing pipeline as training: clean_text then clean_prompt
    # on the prompt; clean_text only on each option.
    prompt_clean = clean_prompt(clean_text(prompt))

    texts = [
        f"{prompt_clean} [SEP] {clean_text(options[label])}"
        for label in labels_present
    ]

    sequences = _tokenizer.texts_to_sequences(texts)

    padded = []

    for seq in sequences:
        seq = seq[:MAX_LENGTH]          # Truncate
        seq = seq + [0] * (MAX_LENGTH - len(seq))  # Pad
        padded.append(seq)

    input_ids = torch.tensor(padded, dtype=torch.long).to(_device)

    with torch.no_grad():
        logits = _model(input_ids)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(this option is correct)

    scores = list(zip(labels_present, probs.cpu().numpy().tolist()))
    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores


def predict_top_k(prompt: str, options: dict, k: int = 3):
    return predict(prompt, options)[:k]
