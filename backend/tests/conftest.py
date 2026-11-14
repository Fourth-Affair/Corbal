import pytest


@pytest.fixture(autouse=True)
def _no_gemini(monkeypatch):
    """Tests never call Gemini: tests of the agent set a key and a fake client themselves."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
