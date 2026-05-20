# Sync thật: `weights/detection` ↔ MLAir registry

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

## Chạy tay

```bash
curl -X POST http://localhost:8000/api/v1/registry/models/sync-full \
  -H "Authorization: Bearer ..."   # không bắt buộc nếu endpoint public
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

`docker compose` đã có service `mlair-artifact-init` (chown volume trước khi `api` start). Stack cũ: chạy script trên hoặc `docker compose up -d mlair-artifact-init api`.

Sau import thành công, trên Hub mở model → tab **Versions** (vd. v1, stage `production`).

## Lưu ý

- Thư mục `v1`, `v2` (lịch sử) **không** bị xóa; chỉ `base` + `production` được căn theo Hub production.
- Đổi file `base` local → vòng sync sau sẽ **push** lên Hub rồi **pull** lại (Hub production thắng nếu train tạo version mới hơn ngay sau đó).
- Sau **train MLAir**, `sync_after_training` căn `base` = bản vừa promote.
