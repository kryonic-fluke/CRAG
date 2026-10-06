import os
from typing import Any, Optional

from app.core.config import settings


def get_chat_llm(temperature: float = 0.0) -> Optional[Any]:
    provider = settings.llm_provider.lower().strip()

    if provider == "gemini":
        gemini_key = (
            settings.gemini_api_key
            or settings.google_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if gemini_key and "mock" not in gemini_key.lower():
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    model=settings.gemini_model_name,
                    google_api_key=gemini_key,
                    temperature=temperature,
                )
            except Exception:
                pass

    if provider == "openai":
        openai_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if (
            openai_key
            and openai_key.startswith("sk-")
            and "mock" not in openai_key.lower()
        ):
            try:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=settings.openai_model_name,
                    api_key=openai_key,
                    temperature=temperature,
                )
            except Exception:
                pass

    return None
