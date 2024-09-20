from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    auth0_domain: str = "test"
    auth0_api_audience: str = "test"
    auth0_issuer: str = "test"
    auth0_algorithms: str = "test"
    auth0_client_id: str = "test"
    auth0_client_secret: str = "test"

    auth0_url: str = "test"
    backend_client_id: str = "test"
    backend_client_secret: str = "test"
    token: str = "test"

    model_config = SettingsConfigDict(env_file=".env")

    def jwks_url(self) -> str:
        return f"https://{self.auth0_domain}/.well-known/jwks.json"


@lru_cache()
def get_settings():
    return Settings()
