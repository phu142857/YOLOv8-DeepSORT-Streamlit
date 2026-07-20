# Baseline E2 — Airflow + MLflow (YOLO)

Triển khai nằm trong repo [`mlair-idle-benchmark/baseline`](../../mlair-idle-benchmark/baseline/README-E2-YOLO.md).

Script từng bước lifecycle (dùng chung với MLAir worker):

- `scripts/airflow_mlair_baseline_step.py`

Luồng đối chiếu paper:

```text
MLAir:  split → detect → prepare → train → eval → gate  (một run_id Hub)
Baseline: cùng bước CV, Airflow DAG + MLflow run riêng, dataset version đọc từ MLAir API
```

Xem [`PhuNT_NhatTM_May2026_Paper/docs/evaluation/yolo-local-airflow-mlflow-baseline-prompt.md`](../../PhuNT_NhatTM_May2026_Paper/docs/evaluation/yolo-local-airflow-mlflow-baseline-prompt.md) để ghi kết quả sau khi chạy.
