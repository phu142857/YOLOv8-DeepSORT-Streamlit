# Triển khai YOLO worker + MLAir controller tách VM

Kiến trúc:

```text
VM controller (182)                 VM worker (184)
┌─────────────────────────┐        ┌──────────────────────────┐
│ mlair :8080             │◄─HTTP──│ cv-api :8000             │
│ tenant yolo / yoloVN    │        │ mlair-cv-train-worker    │
│ ML_AIR_TASK_EXECUTION_  │        │ weights/detection/…      │
│   MODE=external         │        │ (train weights local)    │
└─────────────────────────┘        └──────────────────────────┘
```

**Nguyên tắc:** MLAir (182) = control plane (run, registry, lease API). Worker (184) = thực thi train; **weights đọc từ `weights/detection/{tên_model_hub}/`** khi không mount volume controller.

---

## A. Controller (182) — `~/ml-air`

### A.1 `.env`

```bash
ML_AIR_TASK_EXECUTION_MODE=external
ML_AIR_TASK_LEASE_SECONDS=300
MLAIR_IMAGE=ml-air-cv:latest   # sau bước A.2 — KHÔNG mlair rebuild với tag này
```

### A.2 Image MLAir + plugin YOLO

```bash
cd ~/ml-air
mlair build                    # image gốc ml-air:latest

cd ~/YOLOv8-DeepSORT-Streamlit
docker build -f Dockerfile.mlair-cv -t ml-air-cv:latest \
  --build-arg MLAIR_BASE_IMAGE=ml-air:latest .

cd ~/ml-air
mlair start                    # chỉ start, không rebuild
mlair health
```

### A.3 Tenant / project

Hub → đăng ký scope **yolo / yoloVN** (hoặc dùng script bootstrap sau).

---

## B. Worker token — Service Account (bắt buộc)

**Không dùng PAT admin** cho worker (`tenant default` → lease rỗng).

```bash
cd ~/YOLOv8-DeepSORT-Streamlit
MLAIR_URL=http://<IP-182>:8080 \
MLAIR_ADMIN_PASSWORD='...' \
CV_MLAIR_TENANT=yolo \
CV_MLAIR_PROJECT=yoloVN \
  ./scripts/provision-worker-sa.sh
# → CV_MLAIR_TOKEN=...
```

---

## C. Worker (184) — `~/YOLOv8-DeepSORT-Streamlit`

### C.1 `.env`

```bash
cp .env.worker-external.example .env
# Sửa: MLAIR_CONTROLLER_URL, CV_MLAIR_TOKEN (SA ở trên)
```

### C.2 Weights local (train không cần copy volume từ 182)

Tên thư mục **khớp tên model trên Hub** (vd. `yolov8n`):

```bash
mkdir -p weights/detection/yolov8n/base
# COCO pretrained hoặc checkpoint của bạn:
# weights/detection/yolov8n/base/weights.pt
```

Hoặc bootstrap qua cv-api sau khi stack lên:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/registry/models/bootstrap
```

### C.3 Up stack

```bash
docker compose -f compose.worker-external.yaml -p mlair-cv-worker build
docker compose -f compose.worker-external.yaml -p mlair-cv-worker up -d
./scripts/bootstrap-external-controller.sh
```

### C.4 Verify

```bash
docker logs -f mlair-cv-train-worker
# leased task_id=... plugin=cv_yolo_split
```

Lease test (token SA):

```bash
docker exec mlair-cv-train-worker python3 -c "
import os,json,urllib.request
b=os.environ['MLAIR_API_BASE_URL'].rstrip('/')
t=os.environ['MLAIR_WORKER_TOKEN']
req=urllib.request.Request(b+'/v1/tasks/lease',
  data=json.dumps({'worker_id':'probe','capabilities':['cv_yolo_split'],'max_tasks':1}).encode(),
  method='POST',headers={'Content-Type':'application/json','Authorization':'Bearer '+t})
print(urllib.request.urlopen(req,timeout=15).read().decode())
"
```

---

## D. Train end-to-end

1. Import dataset (≥50 ảnh) — Hub hoặc `curl` cv-api `:8000/api/v1/datasets/import-zip`
2. Hub scope **yolo / yoloVN** → **Train**
3. Worker log: `complete ... cv_yolo_train`

---

## Checklist lỗi

| Triệu chứng | Fix |
|-------------|-----|
| `tasks: []` khi lease | SA scope `yolo/yoloVN`, không dùng PAT admin |
| `PLUGIN_NOT_FOUND` | Build `ml-air-cv`, `mlair start` (không rebuild đè tag cv) |
| `artifact not found` (train) | `weights/detection/{hub_model_name}/base/weights.pt` trên 184 |
| `api_not_external_mode` | `ML_AIR_TASK_EXECUTION_MODE=external` trên 182 |

---

## Scope / project mới

- **Train trên Hub:** admin (hoặc user có quyền) — không cần SA mới.
- **Worker lease:** cập nhật scope SA hoặc tạo SA mới + `CV_MLAIR_TOKEN`.
- **Weights:** thêm `weights/detection/{tên_model_mới}/base/weights.pt` trên worker.

---

## Ports

| VM | Service | Port |
|----|---------|------|
| Controller | MLAir | 8080 |
| Worker | cv-api | 8000 |
| Worker | cv-ui | 8501 |
