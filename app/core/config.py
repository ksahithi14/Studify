import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Studify Agent Backend")
    api_prefix: str = os.getenv("API_PREFIX", "/api")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    huggingface_model: str = os.getenv(
        "HUGGINGFACE_MODEL", "HuggingFaceH4/zephyr-7b-beta"
    )
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    collection_name: str = os.getenv("COLLECTION_NAME", "studify_notes")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
    summary_chunk_chars: int = int(os.getenv("SUMMARY_CHUNK_CHARS", "2800"))
    question_source_chars: int = int(os.getenv("QUESTION_SOURCE_CHARS", "5000"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    os.makedirs(settings.chroma_dir, exist_ok=True)
    return settings

