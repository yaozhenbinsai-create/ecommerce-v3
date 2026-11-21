param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$agentPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $agentPython)) { throw '请先运行 setup.ps1 安装项目依赖。' }
Push-Location -LiteralPath $PSScriptRoot
try { & $agentPython -m uvicorn app.main:app --host 127.0.0.1 --port $Port; exit $LASTEXITCODE }
finally { Pop-Location }
