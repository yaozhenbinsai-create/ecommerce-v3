"""Authenticate the existing in-process HTTP tests against a temporary account."""
import base64
import secrets

from fastapi.testclient import TestClient
import pytest

from app.admin_auth import save_password


@pytest.fixture(scope='session', autouse=True)
def isolated_admin(tmp_path_factory):
    password = secrets.token_urlsafe(32)
    token = base64.b64encode(('ADMIN:' + password).encode('utf-8')).decode('ascii')
    original_init = TestClient.__init__

    def initialize(self, *args, **kwargs):
        kwargs['headers'] = {'Authorization': 'Basic ' + token, **(kwargs.get('headers') or {})}
        original_init(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('BOSS_AGENT_DATA', str(tmp_path_factory.mktemp('admin')))
        save_password(password)
        patch.setattr(TestClient, '__init__', initialize)
        yield
