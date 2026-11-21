"""Local ADMIN setup and HTTP Basic authentication for the public source edition."""
import base64
import binascii
import getpass
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import tempfile

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse


def account_path():
    project = Path(__file__).resolve().parents[1]
    return Path(os.environ.get('BOSS_AGENT_DATA', project / 'data')).resolve() / 'admin.json'


def password_hash(password, salt):
    return hashlib.scrypt(password.encode('utf-8'), salt=salt, n=2**14, r=8, p=1, dklen=32)


def save_password(password):
    if not password:
        raise ValueError('请输入管理员密码。')
    salt = secrets.token_bytes(16)
    record = {'username': 'ADMIN', 'salt': salt.hex(), 'hash': password_hash(password, salt).hex()}
    target = account_path()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent, delete=False) as output:
        staging = Path(output.name)
        try:
            json.dump(record, output)
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            output.close()
            staging.unlink(missing_ok=True)
            raise
    try:
        os.replace(staging, target)
    finally:
        staging.unlink(missing_ok=True)


def verify(authorization):
    scheme, _, encoded = authorization.partition(' ')
    if scheme.lower() != 'basic' or len(encoded) > 8192:
        return False
    try:
        login, separator, password = base64.b64decode(encoded, validate=True).decode('utf-8').partition(':')
        if not separator or login != 'ADMIN':
            return False
        record = json.loads(account_path().read_text(encoding='utf-8'))
        return record['username'] == 'ADMIN' and hmac.compare_digest(
            password_hash(password, bytes.fromhex(record['salt'])), bytes.fromhex(record['hash'])
        )
    except (OSError, UnicodeError, binascii.Error, ValueError, KeyError, TypeError):
        return False


async def require_admin(request, call_next):

    if not await run_in_threadpool(verify, request.headers.get('authorization', '')):
        return JSONResponse({'detail': '请使用本机设置的 ADMIN 账号登录。'}, 401,
                            headers={'WWW-Authenticate': 'Basic realm="Administration", charset="UTF-8"',
                                     'Cache-Control': 'no-store'})
    origin = request.headers.get('origin')
    if request.method not in {'GET', 'HEAD', 'OPTIONS'} and origin and origin != str(request.base_url).rstrip('/'):
        return JSONResponse({'detail': '请求来源不匹配。'}, 403)
    return await call_next(request)


if __name__ == '__main__':
    password = getpass.getpass('设置 ADMIN 密码：')
    if not password or password != getpass.getpass('再次输入密码：'):
        raise SystemExit('两次密码须一致且非空，未保存。')
    save_password(password)
    print('ADMIN 密码已保存，可在本机打开工作台。')
