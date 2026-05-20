"""Training trigger client (Phase 3)."""

from shared.settings import settings


class TrainingClient:
    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = base_url or settings.mlair_api_url
        self.token = token or settings.mlair_token

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.token)
