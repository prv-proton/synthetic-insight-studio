from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    db_path: str = "/data/app.db"
    store_raw: bool = False
    k_threshold: int = 10
    llm_provider: str = "ollama"
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2:1.5b"

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)


settings = Settings()
