import os

from geosym_world.parser.provider_config import chat_completions_url, load_project_env


def test_project_env_loads_provider_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEOSYM_OPENROUTER_BASE_URL", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENROUTER_API_KEY=test-key\n"
        "GEOSYM_OPENROUTER_BASE_URL=https://provider.example/v1/\n",
        encoding="utf-8",
    )

    assert load_project_env(env_path) == env_path
    assert os.environ["OPENROUTER_API_KEY"] == "test-key"
    assert chat_completions_url() == "https://provider.example/v1/chat/completions"


def test_project_env_does_not_override_explicit_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOSYM_OPENROUTER_BASE_URL", "https://explicit.example/v1")
    env_path = tmp_path / ".env"
    env_path.write_text(
        "GEOSYM_OPENROUTER_BASE_URL=https://file.example/v1\n",
        encoding="utf-8",
    )

    load_project_env(env_path)

    assert chat_completions_url() == "https://explicit.example/v1/chat/completions"
