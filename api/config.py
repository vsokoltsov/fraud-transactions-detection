from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    storage: str
    storage_path: str

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()