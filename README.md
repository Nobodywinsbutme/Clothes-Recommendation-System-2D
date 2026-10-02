# Fashion Suggest — Group 10 (IT076IU)

Ứng dụng gợi ý trang phục dựa trên DeepFashion: onboarding trắc nghiệm → like/dislike → re-rank theo thời gian thực. Có đăng ký/đăng nhập để lưu thông tin và lịch sử phối đồ.

## Kiến trúc

| Thư mục | Công nghệ | Port | Vai trò |
|---|---|---|---|
| `frontend/` | Next.js 14 + React + TypeScript | 3000 | Giao diện web (mobile-responsive) |
| `backend/` | Spring Boot 3.3 (Java 21), MySQL, JWT | 8080 | API: auth, user, lịch sử phối đồ, gọi AI service |
| `ai-service/` | Python + PyTorch + FastAPI + CLIP | 8000 | Embedding, tìm kiếm/chấm điểm gợi ý |
| MySQL 8.4 | Docker | 3306 | Dữ liệu user / ứng dụng |
| Qdrant | Docker | 6333 | Vector DB cho embedding DeepFashion |
| `data/` | — | — | DeepFashion (git ignore) |

Luồng: `Browser → Next.js → Spring Boot → FastAPI → Qdrant`. Spring Boot lưu dữ liệu người dùng trong MySQL; AI service chỉ lo phần ML/vector.

## Cần cài trước (mỗi thành viên)

- Git, VS Code
- Docker Desktop (Windows: bật WSL2 backend)
- JDK 21 + Maven 3.9+
- Node.js 20 LTS
- Python 3.11 (Windows: cài từ python.org, tick "Add to PATH")

## Chạy nhanh

```bash
# 1. Setup lần đầu
bash scripts/setup.sh            # macOS/Linux/WSL
# .\scripts\setup.ps1            # Windows PowerShell

# 2. Sửa .env nếu cần (mật khẩu DB, JWT_SECRET)

# 3. Bật MySQL + Qdrant
docker compose up -d mysql qdrant
```

Trong VS Code: `Terminal → Run Task → dev: all` (chạy AI, backend, frontend song song), hoặc chạy từng task `ai: run`, `backend: run`, `frontend: dev`.

Kiểm tra:
- http://localhost:3000 → trang chủ hiển thị trạng thái backend + AI service
- http://localhost:8080/api/health
- http://localhost:8000/health và http://localhost:8000/health/qdrant
- http://localhost:6333/dashboard (Qdrant UI)

Chạy toàn bộ trong Docker (không cần cài JDK/Node/Python): `docker compose --profile full up --build`

## API auth (đã có)

- `POST /api/auth/register` `{email, password, displayName}` → `{token, ...}`
- `POST /api/auth/login` `{email, password}` → `{token, ...}`
- `GET /api/auth/me` (header `Authorization: Bearer <token>`)
- `GET/POST /api/history` — lịch sử phối đồ của user hiện tại

## Lưu ý làm việc nhóm Mac + Windows

- `.gitattributes` ép line ending LF; nên chạy `git config core.autocrlf false` trên Windows.
- Clone repo vào thư mục thường (Windows: tránh đường dẫn quá dài/có dấu).
- Không commit `.env` và `data/`.
- Nếu port 3306 bị trùng MySQL cài sẵn, đổi `MYSQL_PORT` trong `.env` (và `DB_URL` khi chạy backend ngoài Docker).

## Gợi ý phân việc theo timeline proposal

- Tuần 1–2: script ingest DeepFashion → `ai-service/`
- Tuần 5–6: onboarding flow (frontend + lưu `styleProfileJson` ở backend)
- Tuần 7–8: nạp embedding CLIP vào Qdrant, API `/recommend` + feedback like/dislike
