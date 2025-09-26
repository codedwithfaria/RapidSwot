from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    PROJECT_NAME: str = "RapidSwot AI Agent"
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Database Configuration
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "rapidswot"
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    
    # Docker Configuration
    DOCKER_SOCKET: str = "unix://var/run/docker.sock"
    SANDBOX_IMAGE: str = "rapidswot-sandbox:latest"
    
    # VNC Configuration
    VNC_PORT_START: int = 5900
    VNC_WS_PORT_START: int = 6000
    
    class Config:
        env_file = ".env"

settings = Settings()