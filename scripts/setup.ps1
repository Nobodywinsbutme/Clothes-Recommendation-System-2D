# Windows PowerShell: .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
if (-not (Test-Path frontend\.env.local)) { Copy-Item frontend\.env.local.example frontend\.env.local }

Write-Host "==> ai-service: tao venv + cai dependencies"
Set-Location ai-service
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Set-Location ..

Write-Host "==> frontend: npm install"
Set-Location frontend
npm install
Set-Location ..

Write-Host "==> Xong. Chay: docker compose up -d mysql qdrant"
