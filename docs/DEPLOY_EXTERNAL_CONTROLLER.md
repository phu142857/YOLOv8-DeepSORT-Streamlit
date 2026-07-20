# Triển khai YOLO worker + MLAir controller tách VM

Kiến trúc:

```text
VM controller (p2-node-1)          VM worker (GPU/CPU)
┌─────────────────────────┐        ┌──────────────────────────┐
│ mlair :8080 (all-in-one)│◄─HTTP──│ cv-api :8000             │
│ tenant yolo / yoloVN    │        │ mlair-cv-train-worker    │
│ ML_AIR_TASK_EXECUTION_  │        │ cv-ui :8501 (optional)   │
│   MODE=external         │        └──────────────────────────┘
└─────────────────────────┘
```

## Bước 1 — Chuẩn bị controller (`~/ml-air`)

### 1.1 Bật external execution

Trong `~/ml-air/.env`:

```bash
ML_AIR_TASK_EXECUTION_MODE=external
ML_AIR_TASK_LEASE_SECONDS=300
```

```bash
cd ~/ml-air
mlair rebuild
```

### 1.2 Cài plugin contract `cv_yolo_*` trên controller

Hub cần validate pipeline — image `ml-air:latest` thuần **chưa có** plugin YOLO. Build image derived từ repo YOLO:

```bash
cd ~/YOLOv8-DeepSORT-Streamlit   # clone cùng VM hoặc copy repo
docker build -f Dockerfile.mlair-cv -t ml-air-cv:latest \
  --build-arg MLAIR_BASE_IMAGE=ml-air:latest .
```

Trên controller `~/ml-air/.env`:

```bash
MLAIR_IMAGE=ml-air-cv:latest
```

```bash
cd ~/ml-air && mlair rebuild
```

### 1.3 Tenant / project (đã có)

```bash
# yolo / yoloVN — verify
curl -sS -H "Authorization: Bearer $TOKEN" \
  "$API/v1/tenants/yolo/projects" | jq .
```

Hub: chọn scope **yolo / yoloVN**.

---

## Bước 2 — Token cho worker

Trên máy reach được controller:

```bash
cd YOLOv8-DeepSORT-Streamlit
MLAIR_URL=http://192.168.120.182:8080 \
MLAIR_ADMIN_PASSWORD='...' \
  ./scripts/provision-cv-token.sh
# → copy CV_MLAIR_TOKEN=... vào .env worker
```

*(Tuỳ chọn production: tạo Service Account trong Hub → Identity với `tasks:lease`, scope `yolo` / `yoloVN`.)*

---

## Bước 3 — Deploy worker VM

```bash
git clone <yolo-repo> ~/YOLOv8-DeepSORT-Streamlit
cd ~/YOLOv8-DeepSORT-Streamlit

cp .env.worker-external.example .env
# Sửa: MLAIR_CONTROLLER_URL, CV_MLAIR_TOKEN, CV_MLAIR_TENANT, CV_MLAIR_PROJECT

docker compose -f compose.worker-external.yaml build
docker compose -f compose.worker-external.yaml up -d
```

GPU:

```bash
# .env
COMPOSE_FILE=compose.worker-external.yaml:docker-compose.gpu.yml
CV_WORKER_IMAGE=cv-lifecycle-workload:gpu
CV_MLAIR_TRAIN_DEVICE=auto
```

### Bootstrap pipeline + models

```bash
./scripts/bootstrap-external-controller.sh
```

### Verify worker lease

```bash
docker logs -f mlair-cv-train-worker
# Kỳ vọng: cv_yolo_* worker started id=...
```

Trên controller — task `QUEUED` → `RUNNING` khi trigger pipeline từ Hub.

---

## Bước 4 — Artifact sharing (khuyến nghị)

Worker và controller cần đọc cùng dataset/model artifacts khi train:

| Cách | Ghi chú |
|------|---------|
| **NFS** | Export volume `mlair_dataset_artifacts` / `mlair_model_artifacts` từ controller, mount trên worker |
| **API only** | `CV_MLAIR_PERSIST_INGEST_FRAMES=0` (mặc định) — frame qua `cv-api` HTTP + job volume chung trên worker |

Dev nhanh (1 VM): dùng full stack `compose.yaml` (MLAir embedded).

---

## Checklist lỗi thường gặp

| Triệu chứng | Fix |
|-------------|-----|
| Worker `api_not_external_mode` | Controller: `ML_AIR_TASK_EXECUTION_MODE=external` + restart |
| `401` lease | `CV_MLAIR_TOKEN` sai/hết hạn — chạy lại `provision-cv-token.sh` |
| Pipeline không tạo được | Controller thiếu `ml-air-cv` image — bước 1.2 |
| Task `QUEUED` mãi | Worker down hoặc `MLAIR_CAPABILITIES` không khớp plugin |
| Hub scope sai | Chọn **yolo / yoloVN** trên controller |

---

## Port summary

| VM | Service | Port |
|----|---------|------|
| Controller | MLAir Hub/API | 8080 |
| Worker | cv-api | 8000 |
| Worker | cv-ui (Streamlit) | 8501 |
