from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings powered by Pydantic Settings.

    Why this is better than os.getenv():
    1. Automatic type conversion (e.g. '8000' -> int 8000, 'True' -> bool True).
    2. Missing required values trigger explicit, early startup errors.
    3. Seamlessly reads from .env file or system environment variables.
    """

    openai_api_key: str = Field(
        default="mock-key-for-local-testing", description="API key for OpenAI models"
    )
    openai_model_name: str = Field(
        default="gpt-4o-mini", description="Default LLM model identifier"
    )

    tavily_api_key: Optional[str] = Field(
        default=None, description="API key for Tavily Web Search fallback"
    )

    chunk_size: int = Field(
        default=500, ge=50, le=4000, description="Max character size per document chunk"
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
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
