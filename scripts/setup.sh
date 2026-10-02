#!/usr/bin/env bash
# macOS / Linux / WSL: bash scripts/setup.sh
set -e
cd "$(dirname "$0")/.."
[ -f .env ] || cp .env.example .env
[ -f frontend/.env.local ] || cp frontend/.env.local.example frontend/.env.local

echo "==> ai-service: tạo venv + cài dependencies"
cd ai-service
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch          # Mac: bản mặc định; Linux có GPU: xem pytorch.org
.venv/bin/pip install -r requirements.txt
cd ..

echo "==> frontend: npm install"
(cd frontend && npm install)

echo "==> Xong. Chạy: docker compose up -d mysql qdrant"
