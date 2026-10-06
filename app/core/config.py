from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = Field(
        default="gemini",
        description="LLM provider: 'gemini' or 'openai'",
    )

    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key",
    )
    google_api_key: Optional[str] = Field(
        default=None,
        description="Alternative Google API key environment variable",
    )
    gemini_model_name: str = Field(
        default="gemini-2.5-flash",
        description="Default Gemini model identifier",
    )

    openai_api_key: str = Field(
        default="mock-key-for-local-testing",
        description="API key for OpenAI models",
    )
    openai_model_name: str = Field(
        default="gpt-4o-mini",
        description="Default OpenAI model identifier",
    )

    tavily_api_key: Optional[str] = Field(
        default=None,
        description="API key for Tavily Web Search fallback",
    )

    chunk_size: int = Field(
        default=500,
        ge=50,
        le=4000,
        description="Max character size per document chunk",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=1000,
        description="Character overlap between consecutive chunks",
    )

    chroma_persist_directory: str = Field(
        default="data/chroma_db",
        description="Directory path for persistent ChromaDB storage",
    )

    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="API server port")
    debug: bool = Field(default=True, description="Enable debug mode")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
