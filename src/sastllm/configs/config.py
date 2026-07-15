from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration settings.
    Loaded from environment variables or a .env file.
    """

    # API settings
    google_api_key: str = ""
    openai_api_key: str = ""

    # Database settings
    postgres_user: str = "user"
    postgres_password: str = "password"
    postgres_db: str = "database"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_timeout: int = 100000

    # Issel credentials
    endpoint_url: str = "http://localhost:8000"
    access_token: str = "access_key"

    class Config:
        env_file = ".env"


settings = Settings()
