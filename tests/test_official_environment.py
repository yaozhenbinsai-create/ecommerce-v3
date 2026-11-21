import secrets
from pathlib import Path

import yaml

from app.codex_client import CodexAppServer
from app.env_store import EnvStore
from app.research_runtime import ResearchService


def test_fresh_official_login_and_reader_have_no_host_configuration(tmp_path, monkeypatch):
    other = tmp_path / 'different-client'
    other.mkdir()
    (other / 'config.toml').write_text('model_provider = "unrelated"')
    monkeypatch.setenv('CODEX_HOME', str(other))
    monkeypatch.setenv('OPENAI_API_KEY', secrets.token_hex(24))
    monkeypatch.setenv('HTTPS_PROXY', 'https://unused.example.invalid')
    env = CodexAppServer(tmp_path)._process_environment()
    assert Path(env['CODEX_HOME']) == (tmp_path / 'data' / 'official-client').resolve()
    assert list(Path(env['CODEX_HOME']).iterdir()) == []
    assert 'OPENAI_API_KEY' not in env and 'HTTPS_PROXY' not in env
    settings = yaml.safe_load((Path(__file__).parents[1] / 'config' / 'settings.yaml').read_text(encoding='utf-8'))
    assert ResearchService(None, settings).reader_fallback_endpoint == ''


def test_model_secret_uses_the_deployments_private_file(tmp_path, monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY_EXISTING', secrets.token_urlsafe(24))
    (tmp_path / 'config.env').write_text('MODEL_API_KEY_EXISTING=' + secrets.token_urlsafe(24))
    store = EnvStore(tmp_path / 'model-connections.env')
    assert store.get('MODEL_API_KEY_EXISTING') is None
    supplied = secrets.token_urlsafe(24)
    reference = store.create_secret(supplied)
    monkeypatch.setenv(reference, secrets.token_urlsafe(24))
    assert store.get(reference) == supplied
