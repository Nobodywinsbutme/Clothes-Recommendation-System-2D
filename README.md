# Fashion Suggest — Group 10

Đồ án Software Engineering (IT076IU): website gợi ý trang phục từ catalog DeepFashion cố định, học sở thích qua onboarding và like/dislike, không cần lịch sử mua hàng.

**Trạng thái ngày 18/11/2026:** đã có khung ba service, mã auth JWT, API lịch sử outfit và seed 20.000 món với embedding CLIP. **Chưa có luồng recommendation hoàn chỉnh**, API recommendation, onboarding hoặc feedback. Mockup trong báo cáo là thiết kế mục tiêu, chưa phải giao diện đã triển khai.

Đã bổ sung **code thí nghiệm offline** cho hai nhánh CLIP + Qdrant/Rocchio và CLIP + POG, script train/evaluate, kiểm tra dữ liệu và unit tests. Chưa có dataset outfit/history, checkpoint huấn luyện trên dữ liệu thật hoặc kết quả so sánh chất lượng. Xem [hướng dẫn model experiments](docs/model-experiments.md).

## Phạm vi đã thống nhất

- Bản đầu: gợi ý **áo + quần/chân váy hoặc đầm**; giày và phụ kiện làm sau.
- Like/dislike **từng món và cả bộ**; lưu sở thích để dùng lại sau khi đăng nhập lần khác. Chưa có code xử lý các chức năng này.
- Khi dislike cả bộ, hỏi lý do/món cần đổi và chờ user xác nhận trước khi thay gợi ý; không tự coi mọi món trong outfit là bị dislike.
- So sánh **CLIP + Qdrant + cập nhật vector** với **CLIP + POG**. Nhánh POG dùng CLIP features và kiến trúc thí nghiệm lấy cảm hứng từ POG, chưa phải tái lập bài gốc.
- Frontend theo Figma sẽ làm sau khi chủ dự án hoàn thiện thiết kế.
- Theo proposal/midterm: không sinh ảnh bằng AI, không virtual try-on, không kết nối kho hàng thương mại điện tử thời gian thực.
- Public outfit, admin dashboard và moderation có trong midterm nhưng chưa có mã; mức bắt buộc chưa chốt.

## Tiến độ thực tế

| Hạng mục | Đã có | Còn thiếu |
|---|---|---|
| Phân tích, thiết kế | Proposal, midterm, use case, swimlane, mockup | Chốt các điểm chưa thống nhất với phạm vi hiện tại |
| Hạ tầng | Compose MySQL/Qdrant/ba ứng dụng, script setup, VS Code tasks | Xác minh build và chạy tích hợp trên máy nhóm |
| Xử lý dữ liệu | Đọc nhãn, chọn mẫu theo category, crop bbox, CLIP embedding, suy màu | Kiểm tra chất lượng; chưa dùng split train/val/test |
| Seed | 20.000 món và ID riêng, vector float16 kích thước 20.000 × 512 | Xác nhận collection Qdrant trên máy chạy chứa đúng seed |
| Nạp vector | Script upsert, cosine collection, payload và index | Chưa có endpoint truy xuất phục vụ recommendation |
| AI service | Health, embedding; module offline Rocchio/dense retrieval/Qdrant adapter và CLIP+POG | Dữ liệu huấn luyện thật, xác minh Qdrant adapter, API và tích hợp profile/feedback |
| Backend | JWT/BCrypt, User, OutfitHistory, GET/POST history | Profile, session, recommendation, feedback, sở thích dài hạn, item/ảnh |
| Frontend | Trang trạng thái, login/register, lưu token localStorage | Gửi Bearer token, gọi me/history, onboarding, suggestion, wardrobe, Figma |
| Đánh giá | Unit tests model/baseline, synthetic smoke test, công cụ next-item retrieval | Nhãn thật, đo chất lượng outfit, feedback nhiều vòng, latency và test tích hợp |

Theo timeline proposal: dữ liệu và kiến trúc đã có; onboarding thuộc tuần 5–6 còn thiếu; recommendation/feedback tuần 7–8 mới có nền tảng thí nghiệm offline. Chưa hoàn thành tích hợp và đánh giá tuần 9–12. Không quy đổi thành phần trăm vì chưa có trọng số và tiêu chí nghiệm thu.

## Kiến trúc

| Thành phần | Công nghệ/cấu hình trong repo | Cổng | Vai trò |
|---|---|---|---|
| Frontend | Next.js 14.2.18, React 18, TypeScript | 3000 | Giao diện, gọi backend |
| Backend | Spring Boot 3.3.5, Java 21, JPA, JWT | 8080 | Auth, dữ liệu user, lịch sử; dự kiến điều phối AI |
| AI service | Python 3.14 trong Dockerfile, FastAPI, PyTorch, open_clip | 8000 | Hiện có text embedding; dự kiến tìm/chấm điểm gợi ý |
| MySQL | 8.4 | 3306 | Tài khoản, lịch sử; dự kiến thêm profile/session/feedback |
| Qdrant | Server Docker v1.12.4 | 6333 HTTP, 6334 gRPC | Vector và metadata món đồ |

Luồng mục tiêu: **Browser → Next.js → Spring Boot → FastAPI → Qdrant**; Spring Boot lưu dữ liệu ứng dụng trong MySQL. Hiện backend chỉ gọi AI qua `/api/ping-ai`, chưa điều phối recommendation. Frontend hiện chỉ gọi backend.

## Dữ liệu và mô hình

Nguồn: DeepFashion **Category and Attribute Prediction Benchmark**. Nhãn tại máy khai báo 289.222 ảnh, 50 tên category và 1.000 attribute. Script dùng `Anno_coarse`, chưa dùng `Anno_fine`, landmarks hoặc evaluation split.

Pipeline đã viết:

```text
Ảnh + nhãn category/attribute/bbox
    → lấy mẫu theo category, seed=42
    → crop bbox trong bộ nhớ
    → CLIP ViT-B-32 / pretrained=openai
    → chuẩn hóa embedding ảnh, suy màu bằng 12 text prompt
    → seed/items.jsonl + vectors.npy + meta.json
    → load_qdrant.py → collection fashion_items
```

Seed được kiểm tra ngày 18/11/2026:

| Thuộc tính | Giá trị |
|---|---|
| Số món / số ID riêng | 20.000 / 20.000 |
| Vector | 512 chiều, float16, không NaN/Infinity |
| Độ dài vector | Khoảng 0,999785–1,000220; xấp xỉ 1 sau lưu float16 |
| Nhóm nhãn | upper: 9.391; lower: 7.117; full: 3.492 |
| Category xuất hiện | 46 |
| Màu suy đoán | 12 |
| Món không có attribute dương tính | 1.228 |

Payload mỗi món gồm `id`, `image`, `category`, `category_type`, `attributes`, `color`. Màu là suy đoán zero-shot, chưa được xác minh bằng nhãn màu thật. Model embedding hiện tại là CLIP `ViT-B-32/openai`, chưa phải FashionCLIP; script train CLIP+POG đã có nhưng chưa huấn luyện trên dữ liệu thật, không fine-tune CLIP.

### Output mong muốn

Ảnh tham chiếu: [sample.jpg](frontend/public/img/sample.jpg), một outfit board/collage ghép nhiều món trên cùng nền.

- Bản đầu chọn áo + quần/chân váy hoặc đầm từ catalog. Seed chưa có giày/phụ kiện.
- Chọn đồ hợp gu, kiểm tra các món hợp nhau và dàn ảnh thành collage là các bước riêng. Cosine similarity CLIP chưa đủ để khẳng định cả outfit phối đẹp.
- `category_type=full` có cả Coat, Kimono, Robe…; cần ánh xạ category sang vai trò outfit, không coi mọi món full là đầm.
- Crop bbox chỉ dùng cho embedding; script chưa lưu ảnh crop, chưa ghi bbox vào seed, chưa tách nền.
- `image: "img/...jpg"` là đường dẫn trong `data/`, chưa phải URL trình duyệt truy cập được. Chưa có API phục vụ ảnh; Compose cũng chưa mount/serve `data/img`.
- Để ra collage gần ảnh mẫu, cần thêm asset ảnh phù hợp và phần bố trí ảnh; không cần sinh ảnh bằng AI.

### Hướng recommendation dự kiến

Tài liệu giải thích nội bộ đề xuất content-based retrieval bằng CLIP + Qdrant, cập nhật vector kiểu Rocchio:

```text
q = normalize(CLIP_text(mô tả từ onboarding))
q_new = normalize(α·q + β·mean(liked_vectors) − γ·mean(disliked_vectors))
```

**Đã có module thí nghiệm, chưa có API/luồng sản phẩm.** Rocchio, slot mapping, dense baseline và adapter Qdrant nằm trong `ai-service/app/recommendation/baseline.py`; CLIP+POG nằm trong `clip_pog.py`. Cần dữ liệu outfit/history thật, prompt, hệ số, tích hợp feedback có lý do và profile dài hạn. Không mặc định dislike cả bộ là dislike tất cả món.

Các tài liệu H&M/3D vẫn là tham khảo. Nhánh CLIP+POG mới là adaptation có ghi rõ giới hạn trong `docs/model-experiments.md`. `ai-service/app/outfit_samples.py` cũ vẫn lặp lại CLIP loader, chưa có hàm tạo outfit samples như guide mô tả; code dữ liệu mới nằm trong `recommendation/data.py`. Các PDF FashionCLIP/CLIPScore không có nghĩa dự án đã tích hợp các mô hình/chỉ số đó.

## API đã có trong mã nguồn

Đã đối chiếu controller/decorator và security. Đây không phải xác nhận tất cả endpoint đang chạy thành công trên máy hiện tại.

### Backend — http://localhost:8080

| Method | Endpoint | Xác thực hiện tại | Chức năng |
|---|---|---|---|
| GET | /api/health | Public | Trả service=backend, status=ok |
| GET | /api/ping-ai | Public | Gọi AI /health; trả body AI dạng chuỗi hoặc lỗi unreachable |
| POST | /api/auth/register | Public | Tạo user, băm mật khẩu, trả JWT; thành công 201 |
| POST | /api/auth/login | Public | Kiểm tra mật khẩu, trả JWT |
| GET | /api/auth/me | **Cần sửa matcher** | Controller cần Authentication nhưng security đang permit /api/auth/** |
| GET | /api/history | Bearer JWT | Lịch sử user, mới nhất trước |
| POST | /api/history | Bearer JWT | Lưu tên và danh sách ID outfit; thành công 201 |
| GET | /actuator/health | Public | Health do Spring Boot Actuator cung cấp |

Ví dụ đăng ký:

```json
{"email":"demo@example.com","password":"demo-password","displayName":"Demo"}
```

Đăng nhập dùng `{email, password}`. Hai API trả `{token, email, displayName}`; `/me` khi có token hợp lệ trả `{id, email, displayName}`. Register kiểm tra email, password 8–72 ký tự, displayName tối đa 100 ký tự; email trùng trả 409, login sai trả 401. JWT mặc định hết hạn sau 24 giờ.

Ví dụ POST history (ID chỉ minh họa, phải thay bằng món thực và hợp lệ cho outfit):

```json
{"title":"Casual outfit","itemIds":["13","18"]}
```

History response: `{id, title, itemIds, createdAt}`; GET trả một mảng. Hiện chỉ kiểm tra itemIds không rỗng, chưa xác minh item tồn tại hoặc tạo bộ đồ hợp lệ. ID lưu thành chuỗi phân cách dấu phẩy. Chưa trả metadata/ảnh, chưa có sửa/xóa outfit hay phân trang.

### AI service — http://localhost:8000

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | /health | Process AI hoạt động; không xác nhận model đã tải |
| GET | /health/qdrant | Danh sách collection hoặc lỗi kết nối |
| POST | /embed/text | Nhúng `{"text":"casual outfit"}`; trả `{dim, device, embedding}` |

AI service hiện không có auth. FastAPI mặc định cung cấp `/docs`, `/redoc`, `/openapi.json`. Chưa có API embed/upload ảnh; embedding ảnh nằm trong script.

**Chưa có:** API onboarding/profile, session, recommendation, feedback, item detail/ảnh, public outfit, admin, refresh token. `styleProfileJson` mới là cột chờ trong User, chưa có API đọc/ghi.

## Chạy dự án

Chạy các lệnh từ `fashion-suggest/`. Đợt rà soát này chưa build Docker hoặc kiểm thử tích hợp toàn hệ thống.

### 1. Cấu hình và database

```powershell
# Chỉ copy khi chưa có cấu hình riêng
if (!(Test-Path .env)) { Copy-Item .env.example .env }
if (!(Test-Path frontend/.env.local)) { Copy-Item frontend/.env.local.example frontend/.env.local }
docker compose up -d mysql qdrant
```

macOS/Linux copy tương ứng bằng `cp` nếu file chưa tồn tại. Thay các giá trị mẫu; JWT secret cần ít nhất 32 byte trong cấu hình hiện tại. Không commit .env, token hoặc dữ liệu tài khoản thật.

**Môi trường:** .env gốc được Docker Compose dùng để thay biến. Backend chạy Maven chưa tự đọc .env gốc: export biến hoặc dùng mặc định trong application.yml. AI dùng os.getenv, chưa gọi load_dotenv. Frontend đọc frontend/.env.local. Đổi MYSQL_PORT thì phải cập nhật DB_URL cho backend local; đổi cổng Qdrant thì cập nhật QDRANT_URL và đối số --qdrant-url của loader.

### 2. AI service

Dockerfile hiện dùng Python 3.14. Cần xác nhận cài được phiên bản requirements và PyTorch tương ứng trên máy chạy.

```powershell
cd ai-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Ví dụ bản CPU:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

macOS/Linux kích hoạt bằng `source .venv/bin/activate`. Với CUDA/MPS, chọn PyTorch phù hợp máy. Mã tự chọn cuda → mps → cpu; export DEVICE để chỉ định. Lần gọi CLIP đầu có thể tải trọng số, không đại diện độ trễ sau warm-up.

Có script setup tại scripts/setup.ps1 và scripts/setup.sh; bản Windows cài torch CPU. Không cần chạy lại nếu đã có môi trường hoạt động.

### 3. Nạp seed

Khi Qdrant sẵn sàng, từ ai-service với môi trường có numpy và qdrant-client:

```powershell
python scripts/load_qdrant.py
```

Mặc định tạo collection nếu chưa có và upsert theo ID. **--recreate xóa collection cũ rồi tạo lại**, chỉ dùng khi chủ đích thay catalog. Kiểm tra số point và dimension tại http://localhost:6333/dashboard.

Đã có seed thì không cần embed lại. Loader không cần ảnh, GPU hoặc CLIP. Hiển thị ảnh trong sản phẩm vẫn cần nguồn ảnh và URL phục vụ ảnh.

### 4. Backend và frontend

Hai terminal riêng, từ thư mục gốc dự án:

```powershell
# JDK 21 + Maven
cd backend
mvn spring-boot:run
```

```powershell
# Dockerfile frontend dùng Node.js 20
cd frontend
npm ci
npm run dev
```

Mở http://localhost:3000; các trang hiện có: /, /login, /register. Có thể mở fashion-suggest trong VS Code rồi chạy task tương ứng sau khi DB sẵn sàng. Task dev: all chạy song song, chưa đợi MySQL ready khi chạy local.

### 5. Docker toàn bộ

```powershell
docker compose --profile full up --build
```

Cấu hình đã có, chưa xác minh build trong đợt rà soát. AI Dockerfile cài torch từ CPU index, không có GPU passthrough. Compose chưa tự nạp seed, serve ảnh hoặc khai báo volume cache CLIP; nạp seed từ host như bước 3.

## Tạo lại seed khi cần

Script đọc cấu trúc sau, **không phải data/raw/ như data/README.md cũ**:

```text
data/
├── img/                                  # ảnh gốc, không commit
└── Category and Attribute Prediction Benchmark/
    ├── Anno_coarse/
    │   ├── list_category_cloth.txt
    │   ├── list_category_img.txt
    │   ├── list_attr_cloth.txt
    │   ├── list_attr_img.txt
    │   └── list_bbox.txt
    └── Eval/list_eval_partition.txt      # hiện chưa dùng
```

Từ ai-service:

```powershell
python scripts/build_dataset.py --dry-run --limit 2000
# Chỉ khi muốn ghi đè seed:
python scripts/build_dataset.py --limit 20000
```

Dry-run đọc nhãn và kiểm tra tồn tại tối đa 200 đường dẫn ảnh đầu, chưa kiểm tra chất lượng toàn bộ ảnh. --limit 0 lấy toàn bộ dữ liệu. Build ghi lại ba file seed; model truy vấn phải đồng nhất với model tạo vector catalog. Theo README đi kèm dataset, dữ liệu chỉ dùng cho nghiên cứu/giáo dục phi thương mại.

## Lộ trình tiếp theo

1. **Chốt contract bản đầu:** slot áo/quần/đầm, câu hỏi, feedback từng món/cả bộ, profile dài hạn và ngữ cảnh phiên. Chờ Figma hoàn thiện trước khi làm giao diện.
2. **Xác minh nền tảng:** chạy DB, đối chiếu 20.000 point Qdrant, gọi embedding/auth/history. Sửa bảo vệ /api/auth/me, tự gắn Bearer token và chuẩn hóa lỗi.
3. **Catalog và ảnh:** ánh xạ category sang slot; cung cấp metadata/URL ảnh. Chọn ảnh gốc/crop cho bản đầu; tạo asset crop/tách nền riêng nếu cần. Chưa cần embed lại để bắt đầu retrieval.
4. **Recommendation baseline:** onboarding → prompt → CLIP → Qdrant theo slot → ghép outfit; loại trùng/món đã từ chối, xử lý thiếu ứng viên. Trả item ID, slot, URL ảnh, điểm; thêm API AI và backend điều phối.
5. **Profile và session:** API lưu/đọc onboarding, ngữ cảnh phiên, sở thích dài hạn, phiên bản model. Tách nhu cầu phiên như “đi tiệc” khỏi gu ổn định.
6. **Feedback:** lưu event từng món/cả bộ, lý do dislike, các món đã hiển thị; cập nhật sở thích/re-rank, tránh lặp. Xác nhận đăng nhập lại vẫn giữ sở thích.
7. **Sản phẩm theo Figma:** onboarding → outfit board → feedback/follow-up → đặt tên → lưu → wardrobe. Mở rộng history để trả đủ dữ liệu hiển thị, kiểm tra ID và quyền sở hữu.
8. **Đánh giá:** so baseline không feedback với có feedback qua nhiều vòng; đo độ phù hợp, hài lòng, tỷ lệ lặp, độ trễ sau warm-up. Midterm đặt mục tiêu dưới 3 giây; hiện chưa đo. Không dùng riêng cosine/CLIPScore để kết luận người dùng thích hơn hoặc outfit đẹp hơn.
9. **Thí nghiệm và mở rộng:** bổ sung dữ liệu outfit/history để train và so CLIP+POG với baseline; thêm giày/phụ kiện, collage tách nền, public/admin theo phạm vi được chốt. FashionCLIP chưa thuộc phần đã triển khai.

Tên endpoint mới sẽ chốt khi thiết kế contract; bảng API phía trên chỉ liệt kê endpoint đã có.

**Tiêu chí demo lõi:** đăng nhập → onboarding → gợi ý áo/quần hoặc đầm từ catalog thật → phản hồi từng món/cả bộ → gợi ý cập nhật → đặt tên/lưu → xem wardrobe → đăng nhập lại vẫn có lịch sử và sở thích. Hiện chưa đạt tiêu chí này.

## Các điểm cần sửa/xác minh

- Security permit /api/auth/** đang bao gồm /me, trong khi controller cần Authentication; request không token có nguy cơ lỗi server thay vì được bảo vệ đúng ý định.
- Frontend lưu JWT nhưng request helper chưa tự gửi token, chưa gọi /me và history.
- /api/ping-ai và /health/qdrant bắt lỗi rồi vẫn trả HTTP 200; health process không xác nhận DB/model/catalog ready.
- Qdrant client pin 1.19.1, server Compose 1.12.4, mã tắt compatibility check. Cần kiểm tra tương thích khi làm truy vấn.
- Schema MySQL dùng Hibernate ddl-auto: update, chưa có migration. Chưa có entity feedback/session/profile vector.
- History chưa phải wardrobe hoàn chỉnh; thiếu dữ liệu ảnh, metadata và kiểm tra catalog.
- Tài liệu HTML cũ có chỗ ghi chưa lưu token; trang login/register hiện đã lưu localStorage. data/README.md cũng cần đối chiếu cấu trúc script thực tế.

## Cấu trúc và tài liệu

```text
fashion-suggest/
├── frontend/       # Next.js, API client, ảnh tham chiếu
├── backend/        # Spring Boot: auth/security/user/history
├── ai-service/
│   ├── app/        # FastAPI, config, CLIP loader
│   └── scripts/    # build_dataset.py, load_qdrant.py
├── seed/           # metadata và embedding đã tạo
├── data/           # DeepFashion local, Git ignore trừ README
├── docs/           # báo cáo và tài liệu tham khảo
├── scripts/        # setup
├── .vscode/        # tasks và cấu hình phát triển
└── docker-compose.yml
```

- [Proposal](docs/Proposal_Group%2010.pdf): mục tiêu, phạm vi, timeline 12 tuần.
- [Midterm report](docs/Midterm_Report_Group10.pdf): yêu cầu, luồng nghiệp vụ, mockup.
- [CLIP + Qdrant khác POG thế nào?](docs/clip-qdrant-vs-pog.html): giải thích JWT/MySQL, so sánh hai hướng, mô phỏng cập nhật vector và luồng hỏi lý do dislike.
- [Model experiments](docs/model-experiments.md): code hai nhánh, contract dataset, train/evaluate và hướng bổ sung dữ liệu.
- [Giải đáp recommendation/database](docs/giai-dap-recommendation-database.html): đề xuất CLIP/Qdrant và cập nhật sở thích.
- [Quy trình](docs/1-quy-trinh-recommendation.html), [vector và xếp hạng](docs/2-chuyen-hoa-vector-va-sap-xep.html), [luồng website](docs/3-cach-hoat-dong-website.html): tài liệu giải thích, cần đối chiếu source khi có khác biệt.

Rà soát: đã đọc hai báo cáo, mã ứng dụng/cấu hình/script, tài liệu giải thích, xem ảnh mẫu và hình mockup/diagram midterm; kiểm tra metadata/vector seed và cú pháp Python. Không duyệt data/img, node_modules, môi trường ảo hoặc build artifacts. Thử kết nối localhost bị timeout từ môi trường rà soát, nên **chưa xác nhận trạng thái chạy của service hoặc collection Qdrant**. Chưa build/kiểm thử tích hợp.
