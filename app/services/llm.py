from functools import lru_cache

from app.core.config import Settings, get_settings


def build_llm(settings: Settings):
    provider = settings.llm_provider.lower().strip()

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or switch LLM_PROVIDER."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
            api_key=settings.openai_api_key,
        )

    if provider == "huggingface":
        if not settings.huggingface_api_key:
            raise RuntimeError(
                "HUGGINGFACE_API_KEY is not set. Add it to .env or switch LLM_PROVIDER."
            )
        from langchain_huggingface import HuggingFaceEndpoint

        return HuggingFaceEndpoint(
            repo_id=settings.huggingface_model,
            huggingfacehub_api_token=settings.huggingface_api_key,
            temperature=settings.temperature,
            max_new_tokens=settings.max_output_tokens,
        )

    raise RuntimeError(
        "Unsupported LLM_PROVIDER. Use 'openai' or 'huggingface' in .env."
    )


@lru_cache(maxsize=1)
def get_llm():
    return build_llm(get_settings())

