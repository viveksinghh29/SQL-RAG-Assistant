"""Centralized application configuration.

Every configurable value in the system is declared here and sourced from
environment variables (via a `.env` file in development, or real env vars
in production/containers). No module outside `app/config` should read
`os.environ` directly — this keeps configuration auditable in one place
and makes it trivial to mock in tests.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings, validated at startup.

    If a required variable is missing or malformed, the application fails
    fast at import time rather than surfacing a confusing error later.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_name: str = "SQL RAG Assistant"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Security / JWT ---
    jwt_secret_key: str = Field(..., min_length=16)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # --- CORS ---
    cors_allowed_origins: str = "http://localhost:8501"

    # --- Rate limiting ---
    rate_limit_requests_per_minute: int = 60

    # --- Application database (users, chat history, feedback) ---
    app_db_driver: str = "mysql+aiomysql"
    app_db_host: str = "localhost"
    app_db_port: int = 3306
    app_db_user: str = "rag_app_user"
    app_db_password: str = ""
    app_db_name: str = "sql_rag_app_db"

    # --- Target business database (what users query in natural language) ---
    target_db_driver: str = "mysql+aiomysql"
    target_db_host: str = "localhost"
    target_db_port: int = 3306
    target_db_user: str = "rag_readonly_user"
    target_db_password: str = ""
    target_db_name: str = "sample_retail_db"

    # --- LLM ---
    llm_provider: Literal["groq", "openai"] = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_request_timeout_seconds: int = 30

    # --- Embeddings ---
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_device: Literal["cpu", "cuda"] = "cpu"

    # --- Vector store ---
    vector_store_path: str = "./data/vector_store"
    vector_store_top_k: int = 5

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    cache_ttl_seconds: int = 300

    # --- SQL execution limits ---
    sql_query_timeout_seconds: int = 15
    sql_max_rows_returned: int = 1000
    sql_default_limit: int = 100

    @computed_field
    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    @computed_field
    @property
    def app_database_url(self) -> str:
        return (
            f"{self.app_db_driver}://{self.app_db_user}:{self.app_db_password}"
            f"@{self.app_db_host}:{self.app_db_port}/{self.app_db_name}"
        )

    @computed_field
    @property
    def target_database_url(self) -> str:
        return (
            f"{self.target_db_driver}://{self.target_db_user}:{self.target_db_password}"
            f"@{self.target_db_host}:{self.target_db_port}/{self.target_db_name}"
        )

    @computed_field
    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    `lru_cache` ensures the .env file is parsed once per process and the
    same validated object is reused everywhere (and is easy to override
    in tests via `app.config.settings.get_settings.cache_clear()`).
    """
    return Settings()
