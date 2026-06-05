import re
from pathlib import Path
from typing import List


SUBJECT_ALIASES = {
    "RM1": "RMI",
    "RMI": "RMI",
    "RESEARCH METHODOLOGY": "RMI",
    "RESEARCH METHODOLOGY AND IPR": "RMI",
    "DBMS": "DBMS",
    "DATABASE MANAGEMENT SYSTEMS": "DBMS",
    "FDS": "FDS",
    "FOUNDATIONS OF DATA SCIENCE": "FDS",
    "MLG": "MLG",
    "MACHINE LEARNING": "MLG",
    "RAI": "RAI",
    "RESPONSIBLE AI": "RAI",
}


def normalize_subject(subject: str) -> str:
    key = subject.strip().upper()
    return SUBJECT_ALIASES.get(key, key)


def clean_text(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9.,!?;:\s]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def chunk_sentences(sentences: List[str], max_sentences: int = 7) -> List[str]:
    chunks: List[str] = []
    for index in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[index : index + max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_text_by_chars(text: str, max_chars: int = 2800) -> List[str]:
    sentences = split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current and current_length + sentence_length > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_length = sentence_length
        else:
            current.append(sentence)
            current_length += sentence_length + 1

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text[:max_chars].strip()]


def extract_text_from_pdf(pdf_path: str) -> str:
    import fitz

    combined_text = []
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            combined_text.append(page.get_text())
    return clean_text(" ".join(combined_text))


def parse_numbered_lines(text: str) -> List[str]:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-").strip()
        line = re.sub(r"^\d+[\).\s-]+", "", line).strip()
        if line:
            lines.append(line)
    return lines


def safe_filename(name: str) -> str:
    return Path(name).name.replace(" ", "_")

