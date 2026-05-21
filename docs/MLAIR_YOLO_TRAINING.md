# MLAir — train YOLO thật từ dữ liệu user (đúng quy trình framework)

Tài liệu MLAir gốc (đọc trước khi sửa code):

- [Create Plugin](https://github.com/phu142857/ml-air/blob/main/docs/guides/create-plugin.md)
- [Version a Pipeline](https://github.com/phu142857/ml-air/blob/main/docs/guides/version-pipeline.md)
- [Model-centric pipeline mapping & runs/trigger](https://github.com/phu142857/ml-air/blob/main/docs/guides/model-centric-pipeline-mapping-and-trigger.md)
- [HTTP pipeline tasks](https://github.com/phu142857/ml-air/blob/main/docs/guides/http-pipeline-tasks.md)
- [External worker execution](https://github.com/phu142857/ml-air/blob/main/docs/guides/external-worker-execution.md)
- [Configure data readiness](https://github.com/phu142857/ml-air/blob/main/docs/guides/configure-data-readiness-gating.md)

Repo CV **không fork / ghi đè** `ml-air` core. Chỉ cung cấp:

| Thành phần | Đường dẫn |
|------------|-----------|
| Plugin package (cài vào ml-air-api) | `integrations/mlair_cv_plugins/` |
| Pipeline version config (JSON) | `examples/mlair/pipelines/` |
| Train executor (YOLO) | `mlair_adapter/yolo_train_pipeline.py` + `POST /api/v1/mlair/train/execute` |
| External worker (tuỳ chọn) | `scripts/mlair_cv_train_worker.py` |
| Bootstrap API | `scripts/bootstrap_mlair_yolo_pipeline.py` |

## Luồng end-to-end

```text
User Execution (video) → cv pipeline ingest → MLAir buffer → materialize version
    → readiness READY → POST .../runs/trigger
    → pipeline cv-yolo-vehicle-train
    → HTTP task (mặc định) gọi cv-api train
    → Ultralytics fine-tune → import model version lên MLAir
```

## Bước 1 — Cài plugin vào **ml-air-api** (không sửa builtin_reference_plugins)

Trong container hoặc venv của **ml-air-api**:

```bash
pip install -e /path/to/YOLOv8-DeepSORT-Streamlit/integrations/mlair_cv_plugins

curl -X POST "http://localhost:8080/v1/tenants/default/projects/default_project/plugins/reload" \
  -H "Authorization: Bearer admin-token"

curl "http://localhost:8080/v1/tenants/default/projects/default_project/plugins" \
  -H "Authorization: Bearer admin-token"
# phải thấy cv_yolo_train
```

Image tùy chỉnh (additive Dockerfile trong ml-air, không thay `api/Dockerfile`):

```dockerfile
ARG IMG=ghcr.io/phu142857/ml-air-api:latest
FROM ${IMG}
USER root
COPY integrations/mlair_cv_plugins /opt/mlair_cv_plugins
RUN pip install /opt/mlair_cv_plugins
USER appuser
```

(Build context trỏ vào repo CV; COPY đường dẫn package.)

## Hub: "Task plugin is missing in pipeline config"

Hub (`execution-intent-panel`) chỉ bật Train khi **mọi task có `plugin`**. Pipeline version kiểu HTTP (`type: http`) resolve được pipeline nhưng **UI vẫn chặn**.

**Sửa:** đăng ký lại version **plugin** (`cv_yolo_train`) và cài plugin + worker:

```bash
# Sau rebuild cv-api
curl -X POST "http://localhost:8000/api/v1/registry/pipeline/bootstrap"

# Hoặc trực tiếp MLAir (token admin-token)
curl -X POST "http://localhost:8080/v1/tenants/default/projects/default_project/pipelines/cv-yolo-vehicle-train/versions" \
  -H "Authorization: Bearer admin-token" -H "Content-Type: application/json" \
  -d @examples/mlair/pipelines/cv-yolo-vehicle-train.plugin.config.json

docker compose build api && docker compose up -d api scheduler mlair-cv-train-worker cv-api
curl -X POST "http://localhost:8080/v1/tenants/default/projects/default_project/plugins/reload" \
  -H "Authorization: Bearer admin-token"
```

Mặc định stack: `CV_MLAIR_PIPELINE_MODE=plugin`, `ML_AIR_TASK_EXECUTION_MODE=external`, service `mlair-cv-train-worker`.

## Train with model: pipeline unresolved

Hub cần **`PUT .../models/{model_id}/pipeline-mapping`** → `cv-yolo-vehicle-train`. Bootstrap cũ bỏ qua bước này nếu pipeline đã có version.

Sửa nhanh:

```bash
curl -X POST http://localhost:8000/api/v1/registry/pipeline/map-models
# hoặc
curl -X POST http://localhost:8000/api/v1/registry/pipeline/bootstrap
```

Kiểm tra:

```bash
curl -H "Authorization: Bearer admin-token" \
  "http://localhost:8080/v1/tenants/default/projects/default_project/models/<MODEL_ID>/resolved-pipeline"
# pipeline_id phải là cv-yolo-vehicle-train, source: model_pipeline_mapping
```

## Pipeline chỉ 1 task?

Đúng thiết kế: **một task** `yolo_train` (plugin `cv_yolo_train` hoặc HTTP). Ultralytics chạy trong worker/cv-api, không cần DAG nhiều bước.

Trên Hub tab **Topology** sẽ thấy 1 node; inputs vẫn có dataset `cv-traffic-frames`.

## Pipeline không hiện trên Hub

Hub chỉ liệt kê pipeline sau khi có **ít nhất một pipeline version** trong DB (`POST .../pipelines/{id}/versions`). Sync model **không** tự tạo pipeline.

- Mặc định `CV_MLAIR_BOOTSTRAP_PIPELINE=1`: cv-api đăng ký `cv-yolo-vehicle-train` lúc start.
- Tay (sau khi rebuild image có `examples/mlair/`):

```bash
docker exec cv-lifecycle-api python scripts/bootstrap_mlair_yolo_pipeline.py --map-models
```

Kiểm tra:

```bash
curl -sS -H "Authorization: Bearer admin-token" \
  "http://localhost:8080/v1/tenants/default/projects/default_project/pipelines" | jq '.items'
```

## Bước 2 — Đăng ký pipeline version (API MLAir)

Pipeline id: **`cv-yolo-vehicle-train`**

```bash
cd YOLOv8-DeepSORT-Streamlit
CONFIG=$(cat examples/mlair/pipelines/cv-yolo-vehicle-train.http.config.json)

# Validate (run-pipeline.md)
curl -X POST "http://localhost:8080/v1/pipelines/validate" \
  -H "Authorization: Bearer admin-token" \
  -H "Content-Type: application/json" \
  -d "{\"config\": $CONFIG}"

# Publish version (version-pipeline.md)
curl -X POST "http://localhost:8080/v1/tenants/default/projects/default_project/pipelines/cv-yolo-vehicle-train/versions" \
  -H "Authorization: Bearer admin-token" \
  -H "Content-Type: application/json" \
  -d "{\"config\": $CONFIG}"
```

Hoặc một lệnh từ CV:

```bash
docker exec cv-lifecycle-api python scripts/bootstrap_mlair_yolo_pipeline.py --map-models
```

## Bước 3 — Executor HTTP (không sửa mlair_runner)

Trên **ml-air-executor** (đã có trong `docker-compose.yml`):

- `ML_AIR_HTTP_TASK_ALLOWED_HOSTS=cv-api,cv-lifecycle-api`
- `CV_MLAIR_TRAIN_CALLBACK_TOKEN` = token Bearer gọi cv-api

## Bước 4 — Policy, mapping, auto-train CV

```bash
# .env / compose
CV_MLAIR_AUTO_TRAIN=1
CV_MLAIR_MODEL_ID=<uuid-model-trên-hub>
CV_MLAIR_AUTO_PROMOTE=1   # tuỳ chọn
```

Training policy `required_size` phải khớp ngưỡng accumulation trên Hub.

## Bước 5 — Dữ liệu có nhãn

Train dùng **pseudo-labels** từ `detections.jsonl` của từng `job_id` trong manifest. Cần chạy Execution trên video trước khi train.

## Chế độ B — External worker (plugin task)

1. API + scheduler: `ML_AIR_TASK_EXECUTION_MODE=external`
2. Pipeline version: `examples/mlair/pipelines/cv-yolo-vehicle-train.plugin.config.json`
3. Chạy worker:

```bash
MLAIR_API_BASE_URL=http://localhost:8080 \
MLAIR_WORKER_TOKEN=admin-token \
MLAIR_CAPABILITIES=cv_yolo_train \
python scripts/mlair_cv_train_worker.py
```

Worker lease task → gọi `run_yolo_training(plugin_context)` → `POST .../tasks/{id}/complete` với `artifact_uri`.

## Kiểm tra

- Hub: Dataset → Run/Train → Train with model
- `GET .../runs/{run_id}` → SUCCESS
- Model registry có version mới (import từ `best.pt`)
