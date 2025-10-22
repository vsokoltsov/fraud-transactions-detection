from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar

class Settings(BaseSettings):
    storage: str
    model: str
    version: str
    storage_path: str
    features_path: str
    models_path: str
    thresholds_path: str

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env")

settings = Settings() # type: ignore