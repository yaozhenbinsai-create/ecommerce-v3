$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$basePython = (Get-Command py -ErrorAction SilentlyContinue).Source
if (-not $basePython) { $basePython = (Get-Command python -ErrorAction Stop).Source }
& $basePython -m venv (Join-Path $PSScriptRoot '.venv')
if ($LASTEXITCODE -ne 0) { throw 'Python环境创建失败。' }
$dependencyFile = Join-Path $PSScriptRoot 'requirements.lock.txt'
if (-not (Test-Path -LiteralPath $dependencyFile)) { $dependencyFile = Join-Path $PSScriptRoot 'requirements.txt' }
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -m pip install --no-input -r $dependencyFile
exit $LASTEXITCODE
