from functools import lru_cache

from app.core.config import Settings, get_settings


def build_vector_store(settings: Settings):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_dir,
        embedding_function=embeddings,
    )


@lru_cache(maxsize=1)
def get_vector_store():
    return build_vector_store(get_settings())

