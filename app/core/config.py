"""
Application configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Model Configuration
    MODEL_PATH: str = "./models/skin_disease_model.h5"
    MODEL_INPUT_SIZE: int = 224
    MODEL_CLASSES: int = 7
    
    # Data Configuration
    DATA_DIR: str = "./data"
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

