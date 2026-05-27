# MLAir Phase B — Lifecycle pipeline (YOLO)

Pipeline **`cv-yolo-lifecycle-train`** (DAG 4 bước) + **`cv-hard-example-mine`** (tùy chọn).

## Lưu trữ trên container (volume Docker)

| Volume | Mount | Nội dung |
|--------|-------|----------|
| `cv_workload_artifacts` | `/app/artifacts` (cv-api, worker) | jobs, uploads, pipeline state |
| `ml_air_dataset_artifacts` | `/mlair/artifacts/datasets` | frame sau ingest: `cv-runtime-frames/{job_id}/*.jpg` |

Ingest mặc định (`CV_MLAIR_PERSIST_INGEST_FRAMES=0`): frame chỉ ở `artifacts/jobs/{job_id}/frames/`; manifest dùng `http://cv-api/.../frames/`. Train/prepare đọc qua EFS chung hoặc HTTP. Bật `=1` chỉ khi worker không mount `cv-artifacts`.

Copy dữ liệu cũ từ host (một lần, nếu cần):

```bash
docker run --rm -v cv-mlair-stack_cv_workload_artifacts:/dst -v "$(pwd)/artifacts":/src:ro alpine \
  sh -c "cp -a /src/. /dst/ 2>/dev/null || true"
```

## DAG train

```text
prepare (cv_yolo_prepare)
    → train (cv_yolo_train)     # import stage staging
    → eval (cv_yolo_eval)       # mAP50 trên val split
    → gate (cv_yolo_gate)       # so với production; promote nếu pass
```

| Task | Plugin | Việc làm |
|------|--------|----------|
| prepare | `cv_yolo_prepare` | Tải dataset version, pseudo-label từ job detections, `data.yaml` |
| train | `cv_yolo_train` | Ultralytics fine-tune, import `.pt` stage **staging** |
| eval | `cv_yolo_eval` | `val()` checkpoint candidate → `mAP50` |
| gate | `cv_yolo_gate` | `val()` production weights cùng val set; promote nếu `candidate >= prod + delta` |

State giữa các task: `/app/artifacts/mlair_pipeline_runs/{run_id}/state.json` (volume `cv_workload_artifacts`).

## Hard-example mining

Pipeline **`cv-hard-example-mine`** — task `cv_hard_example_mine`:

- Quét dataset version bằng **model production**
- Frame có `max_confidence <= CV_MLAIR_HARD_EXAMPLE_MAX_CONF` → append buffer dataset **`cv-traffic-hard-examples`**

## Cấu hình (`.env`)

```bash
CV_MLAIR_TRAIN_PIPELINE_ID=cv-yolo-lifecycle-train
CV_MLAIR_PIPELINE_MODE=plugin
ML_AIR_TASK_EXECUTION_MODE=external

CV_MLAIR_LIFECYCLE_IMPORT_STAGE=staging
CV_MLAIR_LIFECYCLE_AUTO_PROMOTE=1
CV_MLAIR_GATE_MIN_MAP_DELTA=0.0

CV_MLAIR_HARD_EXAMPLE_PIPELINE_ID=cv-hard-example-mine
CV_MLAIR_HARD_EXAMPLE_DATASET=cv-traffic-hard-examples
CV_MLAIR_HARD_EXAMPLE_MAX_CONF=0.35
CV_MLAIR_HARD_EXAMPLE_MAX_SCAN=500
```

## Triển khai stack

```bash
docker compose build api cv-api
docker compose up -d api scheduler mlair-cv-train-worker cv-api

curl -X POST http://localhost:8000/api/v1/registry/pipeline/bootstrap
curl -X POST "http://localhost:8080/v1/tenants/default/projects/default_project/plugins/reload" \
  -H "Authorization: Bearer admin-token" -d '{}'
```

Worker: `scripts/mlair_cv_pipeline_worker.py` (capabilities = 5 plugin).

API image: `deploy/Dockerfile.mlair-api-cv-plugins` (cài `integrations/mlair_cv_plugins` v0.2).

## Hub — Train with model

1. Model có **pipeline-mapping** → `cv-yolo-lifecycle-train`
2. Dataset version **pinned** + readiness **READY**
3. Run pipeline → 4 task trên worker
4. Gate pass → **promote production** + webhook → `weights/detection`

## Legacy

Pipeline **`cv-yolo-vehicle-train`** (1 task) vẫn dùng được: worker gọi `run_legacy_monolithic_train` khi không có `prepare_ok` trong workspace.

## Train FAILED: `Weights only load failed` / `DetectionModel was not an allowed global`

PyTorch **2.6+** mặc định `torch.load(..., weights_only=True)` — file `.pt` Ultralytics không load được.

- Worker gọi `mlair_adapter.torch_compat.apply_torch_checkpoint_compat()` trước khi import YOLO.
- Rebuild image worker: `docker compose build cv-lifecycle-workload && docker compose up -d mlair-cv-train-worker`
- Chạy **run pipeline mới** (task cũ đã `attempt: 3` / FAILED).

## Hub Run details: Logs / Metrics / Artifacts

| Tab | Nguồn | CV worker |
|-----|--------|-----------|
| **Logs** | `POST /v1/tasks/{id}/logs` → Redis run log stream | `capture_task_logs` |
| **Metrics** | `complete_task` → `run_metrics` (`{plugin}.{key}`) | `metrics` trong body `complete` |
| **Artifacts** | `complete_task` → `run_artifacts` | `artifacts[]` (`train/checkpoint`, `prepare/data.yaml`, …) |

MLAir (bản mới) persist tracking trong `complete_task` / `fail_task`; Hub poll tracking + `run.tracking.updated`.

**Deploy (bắt buộc nếu Hub vẫn trống):** Image `ml-air-api:cv-workload` chỉ **cài plugin** trên base API — base phải là bản ml-air **có** `_persist_run_plugin_tracking` + `POST /tasks/.../logs`.

```bash
# Trong ../ml-air (cùng lệnh bạn đang dùng):
cd ../ml-air
docker build -t ml-air-api:local -f api/Dockerfile .
docker build -t ml-air-scheduler:local -f scheduler/Dockerfile .
docker build -t ml-air-executor:local -f executor/Dockerfile .
docker build -t ml-air-frontend:local -f frontend/Dockerfile .
docker build -t ml-air-realtime:local -f realtime/Dockerfile .

# Trong repo CV — overlay plugin lên ml-air-api:local:
cd ../YOLOv8-DeepSORT-Streamlit
MLAIR_API_IMAGE=ml-air-api:local docker compose build api
docker compose up -d --force-recreate api scheduler executor realtime frontend mlair-cv-train-worker

docker exec ml-air-api python -c "from app.domains.orchestration import worker_task_service as w; print('tracking_ok', hasattr(w,'_persist_run_plugin_tracking'))"
# phải in: tracking_ok True
```

Hoặc một lệnh: `./scripts/build_mlair_local_images.sh` (build 5 image + `api` + CV worker).

Compose mặc định dùng `ml-air-*:local`, `pull_policy: missing` — không kéo GHCR đè bản local.

Sau đó chạy **pipeline run mới** (run cũ không có dữ liệu tracking trong DB).

## Runner logs (Hub) ↔ task CV

MLAir (API + Hub mới) ghi log qua `POST /v1/tasks/{task_id}/logs` → Redis `mlair:logs:{run_id}` + index task. Tab **Runner logs** filter theo task/plugin.

Worker CV (`mlair_cv_pipeline_worker`) tee stdout/stderr + logging Ultralytics vào API đó khi task `RUNNING`.

Cần **ml-air-api:local** (hoặc `cv-workload` build từ base đó) + **ml-air-frontend:local**. Sau đó:

```bash
docker compose build cv-api mlair-cv-train-worker
docker compose up -d api frontend mlair-cv-train-worker
```

Vẫn xem raw: `docker logs -f mlair-cv-train-worker`.

## Train RUNNING mãi / nhảy `train → queued → train`

YOLO train trên CPU có thể **30–60+ phút**. MLAir lease mặc định **30s** — worker **phải heartbeat** (đã thêm trong `mlair_cv_pipeline_worker.py`).

- `ML_AIR_TASK_LEASE_SECONDS=300` trên **api**
- `MLAIR_HEARTBEAT_INTERVAL_SEC=15` trên worker

Nếu run đang kẹt: **Cancel run** trên Hub → rebuild worker → train lại. Kiểm tra log:

```bash
docker logs -f mlair-cv-train-worker
# phải thấy heartbeat + Ultralytics epoch lines
```

Giảm thời gian test: `CV_MLAIR_TRAIN_EPOCHS=3` trong `.env`.

## Lỗi thường gặp: `dataset_version_id is required` trên task prepare

Nguyên nhân: worker cũ chỉ đọc `task.context` top-level; MLAir lease trả **`payload.context`** và **`payload.override_config`**.

Sửa: rebuild/restart `mlair-cv-train-worker` (dùng `mlair_adapter.worker_context.plugin_context_from_lease_task`).

Hub Train: bắt buộc chọn **dataset version** (không chỉ dataset trống) khi `ML_AIR_STRICT_DATASET_VERSION_REQUIRED=1`.

## Lỗi: `no frames downloaded from version ...`

Manifest CSV có `job_id` + `artifact_path` → prepare đọc trực tiếp `artifacts/jobs/{job_id}/frames/` (volume `./artifacts` mount vào worker).

Nếu vẫn lỗi: ingest lúc chạy ngoài Docker có thể ghi `image_uri` = `http://127.0.0.1:8000/...` — worker đã rewrite sang `CV_API_BASE_URL=http://cv-api:8000`. Đảm bảo job frames còn trên disk và `docker compose up -d mlair-cv-train-worker` có mount `./artifacts:/app/artifacts`.

## Test checklist

- [ ] Execution video → buffer +12 rows → materialize → version
- [ ] Train with model → run 4 tasks SUCCESS
- [ ] Registry: version mới ở staging, sau gate → production
- [ ] Inference dùng weights production mới
- [ ] (Tuỳ chọn) Run pipeline `cv-hard-example-mine` trên version → buffer hard-examples tăng
