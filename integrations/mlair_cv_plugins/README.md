# MLAir plugins (CV workload)

Package cài **vào môi trường ml-air-api** theo [MLAir — Create Plugin](https://github.com/phu142857/ml-air/blob/main/docs/guides/create-plugin.md). Không sửa `builtin_reference_plugins` trong repo ml-air.

```bash
pip install -e /path/to/YOLOv8-DeepSORT-Streamlit/integrations/mlair_cv_plugins
curl -X POST "http://localhost:8080/v1/tenants/default/projects/default_project/plugins/reload" \
  -H "Authorization: Bearer admin-token"
```

Chi tiết: [docs/MLAIR_YOLO_TRAINING.md](../../docs/MLAIR_YOLO_TRAINING.md).
