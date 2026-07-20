# Sync thật: `weights/detection` ↔ MLAir registry

Kiến trúc tổng thể (học Vet-AI): [MLAIR_MODEL_ARCHITECTURE.md](./MLAIR_MODEL_ARCHITECTURE.md).

## Mục tiêu

Sau sync, **cùng một file weights** cho mỗi tên model (vd. `yolov8n`):

- `weights/detection/yolov8n/base/weights.pt`
- `weights/detection/yolov8n/production/weights.pt`
- MLAir production artifact (Hub)
- `weights/registry/{model_id}/vN.pt` (cache inference)

`[local] yolov8n / base` và `[MLAir] yolov8n (vN)` dùng **cùng bytes** (không còn hai bản lệch).

## Cơ chế `sync_full`

1. **Push** — với mỗi tên model trên disk, lấy checkpoint chuẩn (`base` → `production` → `vN` cao nhất), nếu đổi so với lần sync trước → `import` lên MLAir.
2. **Pull** — mọi model trên Hub: production → copy vào **cả** `base` và `production` local.
3. Ghi SHA256 vào `artifacts/.mlair_model_sync_state.json`.

Tự chạy khi `CV_MLAIR_AUTO_SYNC_MODELS=1` (mặc định): lúc cv-api start + mỗi 5 phút.

## Build / up xong — tự sync (không cần curl tay)

Dùng một trong các lệnh sau (tự `chown` volume + `sync-full?force=1` + pipeline):

```bash
./scripts/docker-up.sh          # compose up -d + post bootstrap
./scripts/docker-rebuild.sh     # build cv-api + up + bootstrap
./scripts/post_stack_bootstrap.sh   # chỉ bootstrap (stack đã chạy)
```

Tắt bootstrap sau up: `CV_SKIP_POST_BOOTSTRAP=1 docker compose up -d`

## Sync mặc định (tự động)

Khi `CV_MLAIR_AUTO_SYNC_MODELS=1` và `CV_MLAIR_SYNC_ON_STARTUP=1` (mặc định trong `docker-compose.yml`), **cv-api** sau khi MLAir API sẵn sàng sẽ:

1. Dọn state cũ nếu Hub bị xóa (`docker compose down -v`) nhưng file `artifacts/.mlair_model_sync_state.json` trên máy host vẫn còn.
2. Push `weights/detection/*/base/weights.pt` lên Hub.
3. Lặp lại mỗi `CV_MLAIR_SYNC_INTERVAL_SEC` (mặc định 300s).

**Lưu ý:** `down -v` xóa DB/volume Docker, **không** xóa `./artifacts/` trên host — trước đây sync báo `already_aligned` trong khi Hub trống; đã sửa bằng reconcile + force khi Hub rỗng.

## Chạy tay

```bash
curl -X POST "http://localhost:8000/api/v1/registry/models/sync-full"
# Sau khi reset volume Hub mà vẫn báo already_aligned:
curl -X POST "http://localhost:8000/api/v1/registry/models/sync-full?force=1"
```

Hoặc:

```bash
docker exec cv-lifecycle-api python -c "
from mlair_adapter.model_sync import ModelSyncService
print(ModelSyncService().sync_full())
"
```

## Env

```bash
CV_MLAIR_AUTO_SYNC_MODELS=1
CV_MLAIR_SYNC_ON_STARTUP=1
CV_MLAIR_SYNC_INTERVAL_SEC=300
CV_MLAIR_SYNC_AFTER_TRAIN=1   # sau train, pull production → base
```

## Hub không hiện version (model trống)

**Không phải do MLAir không đọc `.pt`** — Hub và API hỗ trợ `.pt` / `.onnx` / … qua `POST .../versions/import` (field `model_file`).

Thường gặp: **import trả 500** vì volume `ml_air_model_artifacts` thuộc `root`, trong khi `ml-air-api` chạy user `appuser` (uid 1000) → không ghi được artifact → DB có model nhưng `versions` rỗng, `production_version: null`.

Kiểm tra:

```bash
curl -sS -X POST http://localhost:8000/api/v1/registry/models/sync-full | jq '.push_results[0]'
docker logs ml-air-api 2>&1 | tail -20   # Permission denied: '/mlair/artifacts/models/...'
```

Sửa quyền rồi sync lại:

```bash
./scripts/fix_mlair_model_artifacts_perm.sh
curl -X POST http://localhost:8000/api/v1/registry/models/sync-full
```

### `cv-mlair-stack-mlair-artifact-init-1` là gì?

Container **chạy một lần** (trạng thái `Exited (0)` là bình thường): `chown` volume `ml_air_model_artifacts` và `ml_air_dataset_artifacts` cho user `appuser` (uid 1000) trước khi `ml-air-api` ghi file `.pt`. Không có init → import version **500 Permission denied** → Hub có **tên model** nhưng **không có version** (`production_version: null`).

`ml-air-api` giờ cũng `chown` lại **mỗi lần start**. Nếu vẫn thiếu version:

```bash
./scripts/fix_mlair_model_artifacts_perm.sh
curl -X POST "http://localhost:8000/api/v1/registry/models/sync-full?force=1"
```

Hoặc `./scripts/docker-up.sh` (tự chạy lại init).

Sau import thành công, trên Hub mở model → tab **Versions** (vd. v1, stage `production`).

## Promote trên Hub → Vehicle Detection dùng đúng bản

Promote version lên **production** trên MLAir Hub sẽ (qua webhook) cập nhật `weights/detection/<model>/base/weights.pt`. UI chọn model dạng `yolov8n/base` tự dùng bản production mới.

Cần trong `docker-compose` (đã mặc định):

- `MLAIR_MODEL_PROMOTE_WEBHOOK_URL=http://cv-api:8000/api/v1/mlair/promote-webhook`
- `MLAIR_MODEL_PROMOTE_WEBHOOK_BEARER_TOKEN` trùng `CV_MLAIR_PROMOTE_WEBHOOK_TOKEN`

Sau `docker compose up -d api cv-api`, promote thử trên Hub và kiểm tra:

```bash
sha256sum weights/detection/yolov8n/base/weights.pt
cat weights/detection/yolov8n/base/mlair-sync.json
```

## Lưu ý

- Thư mục `v1`, `v2` (lịch sử) **không** bị xóa; chỉ `base` + `production` được căn theo Hub production.
- Đổi file `base` local → vòng sync sau sẽ **push** lên Hub rồi **pull** lại (Hub production thắng nếu train tạo version mới hơn ngay sau đó).
- Sau **train MLAir**, `sync_after_training` căn `base` = bản vừa promote.
