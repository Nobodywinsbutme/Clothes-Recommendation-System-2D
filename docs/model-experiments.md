# So sánh CLIP + Rocchio với CLIP + POG

Quyết định ngày 18/11/2026: chuẩn bị hai nhánh **CLIP + Qdrant + cập nhật vector** và **CLIP + POG**. Chưa có dataset outfit/lịch sử user; chưa huấn luyện model trên dữ liệu thật, chưa có kết luận nhánh nào tốt hơn. Không xây thêm một nhánh POG nguyên bản riêng.

## Code hiện đã chuẩn bị

| File | Chức năng |
|---|---|
| `ai-service/app/recommendation/baseline.py` | Rocchio, cosine reference scorer, ánh xạ slot, ghép outfit baseline, Qdrant search adapter |
| `ai-service/app/recommendation/clip_pog.py` | Model encoder/decoder nhận CLIP features, profile token, causal mask, history padding, chọn món theo slot |
| `ai-service/app/recommendation/data.py` | Đọc catalog/manifest, kiểm tra ID, bố cục outfit, thời gian lịch sử/split; tạo training batch |
| `ai-service/app/recommendation/evaluation.py` | So sánh next-item retrieval cùng catalog/profile/candidate pool; Hit@K, NDCG@K, MRR |
| `ai-service/scripts/train_clip_pog.py` | Kiểm tra dữ liệu, train, chọn checkpoint bằng validation loss, evaluate riêng |
| `ai-service/tests/test_recommendation.py` | Kiểm tra masking/chống nhìn target tương lai, feedback, layout, dữ liệu và gradient |

Đây là **mã thí nghiệm offline**, chưa gắn vào FastAPI/Spring Boot hoặc frontend. Module `baseline.py` có adapter Qdrant nhưng adapter chưa được chạy với server thật trong lượt này. Reference scorer tính cosine trực tiếp giúp kiểm soát phép so sánh, không cần database để chạy test.

Nhánh B là **POG-inspired adaptation**, không phải tái lập nguyên bản bài báo: dùng CLIP frozen thay encoder multimodal gốc; thêm profile token; full-catalog cross entropy thay sampled objective; chưa có FOM pretraining. Generation bản đầu bị ràng buộc áo + quần hoặc đầm, dừng theo layout cố định; END được học khi train nhưng chưa quyết định dừng ở chế độ generation này. Điểm của hai model có thang đo khác nhau, không so trực tiếp giá trị score.

## Chạy kiểm tra ngay, không cần dataset mới

Từ `fashion-suggest/ai-service/`, với Python đã cài PyTorch:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/train_clip_pog.py --smoke-test
```

Smoke test dùng tensor giả để kiểm tra forward/backward và generation đúng slot, không cần NumPy/Qdrant/CLIP download. Không lưu checkpoint, không báo số liệu chất lượng. Không dùng model khởi tạo ngẫu nhiên để demo recommendation cho người dùng.

## Dữ liệu còn thiếu

Seed DeepFashion cho biết từng ảnh là món gì; không cho biết “áo X và quần Y là một outfit tốt”, cũng không cung cấp chuỗi sở thích của user. Không tự coi các món có vector gần nhau là nhãn phối đồ chuẩn, rồi dùng chính nhãn đó để chứng minh model tốt hơn baseline.

Các hướng bổ sung:

1. **POG dataset:** repository tác giả có định dạng outfit, user-history/outfit và item. Đây là hướng đáng kiểm tra trước nếu cần cả dữ liệu phối đồ và hành vi. Chưa tải/xác minh ảnh, điều kiện sử dụng hoặc khả năng dùng đầy đủ. Nếu dùng catalog này phải embed các ảnh tương ứng bằng cùng CLIP, giữ namespace ID riêng; không nối ID với DeepFashion theo số thứ tự. [Nguồn tác giả](https://github.com/wenyuer/POG).
2. **Polyvore:** có outfit và tác vụ đánh giá phối đồ; phù hợp nghiên cứu phần tương thích. Không tự xem tác giả của outfit là lịch sử click của một người dùng. Tài liệu dataset cảnh báo URL ảnh gốc đã không còn hoạt động, nên phải kiểm tra nguồn ảnh lưu trữ trước khi chọn. Dataset outfit đơn thuần chưa đủ cho bài toán cá nhân hóa có lịch sử; cần bổ sung dữ liệu hoặc một giai đoạn compatibility-only riêng, hiện chưa viết. [Nguồn tác giả](https://github.com/xthan/polyvore-dataset).
3. **Dữ liệu nhóm tự thu thập trên DeepFashion:** người thật phối/chọn bộ, cung cấp lịch sử sở thích trước đó và đánh giá. Phù hợp catalog demo nhưng tốn công, có thiên lệch; một tập nhỏ chỉ phù hợp pilot, chưa đủ cơ sở đảm bảo Transformer học tốt. Người tạo nhãn nên độc lập với model cần đánh giá.

Nếu dữ liệu ngoài chỉ hỗ trợ thời điểm tương đối hoặc split chính thức, cần viết adapter và chính sách kiểm tra phù hợp trước khi nhập. **Không bịa timestamp/history để qua validator.** Validator hiện dùng split thời gian nghiêm ngặt; dữ liệu Polyvore chia sẵn không mặc định đáp ứng chính sách đó.

## Contract dữ liệu chuẩn hóa

Catalog vẫn dùng `items.jsonl`, `vectors.npy`, `meta.json` theo thứ tự ID đồng nhất. Model/candidate queries phải dùng cùng embedding model. Mỗi dòng samples JSONL có cấu trúc sau (các ID ký hiệu chỉ minh họa, không phải một dataset chạy sẵn):

```json
{
  "sample_id": "observation-001",
  "user_id": "anonymous-user-001",
  "split": "train",
  "timestamp": "2026-01-02T10:00:00Z",
  "history": [
    {"item_id": "TOP_ID_FROM_CATALOG", "timestamp": "2026-01-01T10:00:00Z"}
  ],
  "outfit_item_ids": ["TOP_ID_FROM_CATALOG", "BOTTOM_ID_FROM_CATALOG"]
}
```

- ID luôn là chuỗi, phải tồn tại trong catalog. Dùng user ID giả danh, không nhập email/tên thật.
- `history` là tương tác sở thích dương tính đã có trước thời điểm target, sắp theo thời gian. Dislike cần xử lý thành profile có phạm vi/lý do riêng; chưa có pipeline event → profile tự động.
- Outfit hợp lệ bản đầu: `[top, bottom]` theo thứ tự hoặc `[dress]`. Không suy từ `category_type=full` rằng mọi món là đầm.
- Có thể cung cấp `query_embedding`: vector CLIP 512 chiều từ onboarding/profile đã biết **trước** target, không tạo từ mô tả chứa đáp án. Nếu không có, dùng trung bình vector lịch sử dương tính; đây là proxy có giới hạn.
- User không lịch sử bắt buộc có query embedding. Không lấy trung bình target outfit làm “sở thích đầu vào”.
- Có đủ `train`, `val`, `test`; thời gian train < val < test. Cùng tổ hợp item không được xuất hiện ở hai split. Cần xử lý thêm cùng sản phẩm/ảnh gần trùng ở bước tiền xử lý; validator hiện không phát hiện được tất cả trùng hình.
- Manifest phải ghi được nguồn, phương pháp gán nhãn và chính sách split trong tài liệu đi kèm. Validator chỉ kiểm tra contract, không xác nhận nhãn là thật.

## Chạy khi đã có dataset thật

Chuẩn hóa dữ liệu vào `fashion-suggest/experiments/samples.jsonl` (bị Git ignore), cài các dependency hiện có trong requirements, gồm NumPy. Từ `ai-service/`:

```powershell
python scripts/train_clip_pog.py --samples ../experiments/samples.jsonl --validate-only
python scripts/train_clip_pog.py --samples ../experiments/samples.jsonl --output ../experiments/clip-pog-seed42.pt --seed 42
python scripts/train_clip_pog.py --samples ../experiments/samples.jsonl --checkpoint ../experiments/clip-pog-seed42.pt --split val
# Chỉ mở test sau khi chốt model/hyperparameter:
python scripts/train_clip_pog.py --samples ../experiments/samples.jsonl --checkpoint ../experiments/clip-pog-seed42.pt --split test
```

Script mặc định chạy CPU, có `--device cuda` khi môi trường hỗ trợ. Không ghi đè output có sẵn khi bắt đầu một run mới; trong run checkpoint tốt nhất được cập nhật bằng validation loss. Checkpoint gắn hash catalog và sample manifest để tránh đánh giá nhầm model/vector/split. Dữ liệu có thể có lịch sử lặp lại target item nếu hành vi đó thật sự có trước; việc mua/like lại không tự bị coi là leak.

## So sánh công bằng và giới hạn hiện tại

**Đã viết:** next-item retrieval với teacher forcing. Hai nhánh dùng cùng profile, catalog, slot, lịch sử, các món trước đó trong outfit và bộ ứng viên. B được cấp prefix đúng trước target; A là baseline chỉ dùng profile. Report tên A là `clip_profile_dense`, B là `clip_pog`. Đây là đánh giá phần model thêm ngữ cảnh outfit, chưa phải toàn bộ vòng Rocchio/Qdrant trong sản phẩm.

**Chưa viết/chưa đo:** FITB đầy đủ hai phía, compatibility prediction, user study, replay nhiều vòng feedback, đánh giá profile dài hạn, độ trễ server/Qdrant và chất lượng free-running outfits. Không gọi teacher-forced next-item là FITB hoặc bằng chứng cải thiện sau dislike.

Để kết luận nhánh nào tốt hơn:

1. Khóa catalog, split và nguồn nhãn; dùng cùng bộ ứng viên hoặc công bố riêng ảnh hưởng ANN retrieval.
2. Đánh giá item retrieval trên dữ liệu giữ lại; báo thêm loại layout và kích thước candidate pool. Không so kết quả từ hai catalog khác nhau rồi quy hết chênh lệch cho thuật toán.
3. Cho người đánh giá so sánh outfit A/B mà không biết model nào tạo, ghi riêng hợp gu, hợp dịp và các món hợp nhau. Ghi rõ số người/mẫu và độ bất định; không chọn riêng ví dụ đẹp.
4. Replay cùng chuỗi feedback cho hai nhánh hoặc làm user study có kiểm soát. Không dislike cả bộ rồi tự gán dislike cho từng món.
5. Lặp nhiều seed, báo trung bình/độ phân tán; đo latency sau warm-up trên cùng thiết bị. Test set không dùng để chọn hyperparameter.
6. Nếu B không tốt hơn A hoặc chậm hơn đáng kể thì giữ A cho demo; kết quả đó vẫn hợp lệ. Chưa có phép đo nên chưa tuyên bố nhánh thắng.

## Việc cần làm ngay trong lúc hoàn thiện Figma

- Xác định nguồn outfit/history thực sự tải được và phù hợp sử dụng; đây là nút thắt của nhánh B.
- Chốt câu hỏi onboarding, lý do dislike và cách phân biệt “chỉ phiên này” với “sở thích thường xuyên”. Câu hỏi cần biến được thành profile hoặc filter.
- Có màn hình suggestion cho từng món/cả bộ; khi dislike giữ bộ cũ, hỏi lý do rồi mới xác nhận đổi. Cho phép giữ những món đã thích.
- Đánh dấu trạng thái loading/error/empty/no-candidates, Back/Continue, đổi câu trả lời, đặt tên/lưu outfit và wardrobe. Có màn hình mobile/desktop hoặc ghi rõ có thể tự bổ sung responsive.
- Khi gửi Figma, cung cấp link có quyền xem. Prototype/workflow cùng các frame giao diện là đủ để bắt đầu; màu/font/component/asset cần có nếu muốn bám sát hình thức. Những chi tiết chưa thiết kế có thể được bổ sung theo cùng hệ thống giao diện.

MySQL theo Compose chạy trong container và dùng named volume; không cần bật MySQL service Windows riêng. Kiểm tra Docker hiện không kết nối được Engine; chưa xác nhận container/volume thực tế. Không chạy lệnh xóa volume để xử lý việc Engine chưa khởi động.
