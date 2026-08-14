import json
import random
import unicodedata
import re


TRAINING_DATA_FILENAME = "input_database.jsonl"


def load_dataset() -> dict:
    with open(TRAINING_DATA_FILENAME, "r") as f:
        raw_json_data = [json.loads(line) for line in f]

    text_data = []
    labels_data = []
    label_counts = {0: 0, 1: 0}

    for sample_line in raw_json_data:
        try:
            if sample_line["language"] != "en":
                continue
            sample_text = clean_text(sample_line["text"])
            sample_label_raw = sample_line["is_it_about_ai"]

            if len(sample_text) < 2:
                continue

            if sample_label_raw.lower() == "yes":
                sample_label = 1
                label_counts[1] += 1

            elif sample_label_raw.lower() == "no":
                sample_label = 0
                label_counts[0] += 1

            else:
                raise ValueError(f"Unexpected label value: {sample_label_raw}")

            # if sample_line["is_it_organic"] == "no":
            #     continue

            text_data.append(sample_text)
            labels_data.append(sample_label)
        except KeyError:
            continue

    if len(text_data) != len(labels_data):
        raise ValueError("Mismatch between number of texts and labels.")

    formatted_data = {
        "text": text_data,
        "labels": labels_data
    }

    print(
        f"Loaded {len(formatted_data['text'])} samples from {TRAINING_DATA_FILENAME}. Label counts: {label_counts}")

    return formatted_data


def clean_text(text: str) -> str:
    """
    Strictly normalizes string inputs to prevent multi-byte Unicode errors,
    control-character bugs, and tokenization overflows.

    Contains comments for equivalent JavaScript (for browser deployment).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. NFD Decomposition: Splits accented characters into base letter + accent mark
    # (e.g., 'é' -> 'e' + '´')
    # JS: text = text.normalize('NFD');
    text = unicodedata.normalize('NFD', text)

    # 2. Strip combining diacritical marks (accents/decorations)
    # JS: text = text.replace(/[\u0300-\u036f]/g, '');
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')

    # 3. Strip control characters, invisible spaces, and non-printable bytes (\x00-\x1F, zero-width spaces, etc.)
    # JS: text = text.replace(/[\u0000-\u001F\u007F-\u009F\u200B-\u200D\uFEFF]/g, '');
    text = re.sub(r'[\x00-\x1f\x7f-\x9f\u200b-\u200d\ufeff]', '', text)

    # 4. Enforce strict ASCII range: Strip emojis, foreign scripts, and non-standard symbols
    # Keep standard ASCII alphanumeric characters and basic punctuation: a-z, A-Z, 0-9, spaces, and . , ! ? -
    # JS: text = text.replace(/[^a-zA-Z0-9\s.,!?\-']/g, '');
    # Enclose the raw string in double quotes so the single quote inside is ignored
    text = re.sub(r"[^a-zA-Z0-9\s.,!?\-']", '', text)

    # 5. Normalize all whitespace (tabs, newlines, multi-spaces) down to single spaces
    # JS: text = text.replace(/\s+/g, ' ').trim();
    text = re.sub(r'\s+', ' ', text).strip()

    return text
