"""
E2 baseline DAG — CV lifecycle (same steps as MLAir cv-yolo-lifecycle-train), Airflow + MLflow.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

CV_REPO = os.environ.get("CV_REPO", "/opt/cv")
CV_ARTIFACTS = os.environ.get("CV_ARTIFACT_ROOT", "/opt/cv-artifacts")
CV_IMAGE = os.environ.get("CV_WORKER_IMAGE", "cv-lifecycle-workload:gpu")
MLAIR_API = os.environ.get("CV_MLAIR_API_URL", "http://host.containers.internal:8080")
MLAIR_TOKEN = os.environ.get("CV_MLAIR_TOKEN", "admin-token")
CV_API = os.environ.get("CV_API_BASE_URL", "http://host.containers.internal:8000")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://host.containers.internal:5000")
MLFLOW_EXP = os.environ.get("MLFLOW_EXPERIMENT_NAME", "cv-yolo-lifecycle-baseline-e2")
BASELINE_RUNNER_URL = os.environ.get(
    "BASELINE_RUNNER_URL", "http://baseline-cv-runner:9191"
)
E2_STOP_AFTER = os.environ.get("E2_STOP_AFTER", "prepare")


def _http_step(step: str) -> str:
    return f"""
set -euo pipefail
BODY=$(python3 -c "import json; print(json.dumps({{
  'step': '{step}',
  'mode': '{{{{ dag_run.conf.get('mode', params.mode) }}}}',
  'model_id': '{{{{ dag_run.conf.get('mlair_model_id', params.mlair_model_id) }}}}',
  'dataset_version_id': '{{{{ dag_run.conf.get('mlair_input_version_id', params.mlair_input_version_id) }}}}',
  'train_ready_version_id': '{{{{ dag_run.conf.get('mlair_train_ready_version_id', params.mlair_train_ready_version_id) }}}}',
  'baseline_run_id': 'af-{{{{ dag_run.run_id }}}}',
  'env': {{
    'E2_STOP_AFTER': '{E2_STOP_AFTER}',
    'CV_MLAIR_API_URL': '{MLAIR_API}',
    'CV_MLAIR_TOKEN': '{MLAIR_TOKEN}',
    'CV_API_BASE_URL': '{CV_API}',
    'CV_ARTIFACT_ROOT': '/app/artifacts',
    'BASELINE_RUN_ID': 'af-{{{{ dag_run.run_id }}}}',
    'BASELINE_MODE': '{{{{ dag_run.conf.get('mode', params.mode) }}}}',
    'MLFLOW_TRACKING_URI': '{MLFLOW_URI}',
    'MLFLOW_EXPERIMENT_NAME': '{MLFLOW_EXP}',
  }},
}}))")
CODE=$(curl -sf -o /tmp/step_out.json -w "%{{http_code}}" -X POST "{BASELINE_RUNNER_URL}/step" \\
  -H "Content-Type: application/json" -d "$BODY" || echo "000")
cat /tmp/step_out.json
if [[ "$CODE" != "200" ]]; then exit 1; fi
python3 -c "import json,sys; r=json.load(open('/tmp/step_out.json')); sys.exit(0 if r.get('ok') else 1)"
"""


def _choose_mode_branch(**context) -> str:
    dr = context.get("dag_run")
    conf = (dr.conf if dr and dr.conf else {}) or {}
    params = context.get("params") or {}
    mode = str(conf.get("mode") or params.get("mode") or "train_ready").strip().lower()
    return "cv_split" if mode == "full" else "cv_prepare"


def _after_prepare_branch(**context) -> str:
    if str(os.environ.get("E2_STOP_AFTER", "prepare")).strip().lower() == "prepare":
        return "e2_done"
    return "cv_train"


default_args = {
    "owner": "e2-baseline",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="cv_yolo_mlair_baseline",
    default_args=default_args,
    description="YOLO lifecycle baseline: MLAir datasets + Airflow + MLflow",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["e2", "yolo", "mlair", "mlflow", "baseline"],
    params={
        "mode": Param("train_ready", type="string", enum=["train_ready", "full"]),
        "mlair_model_id": Param("", type="string"),
        "mlair_input_version_id": Param("", type="string"),
        "mlair_train_ready_version_id": Param("", type="string"),
    },
    render_template_as_native_obj=True,
) as dag:
    mode_branch = BranchPythonOperator(
        task_id="mode_branch",
        python_callable=_choose_mode_branch,
    )
    cv_split = BashOperator(task_id="cv_split", bash_command=_http_step("split"))
    cv_detect = BashOperator(task_id="cv_detect", bash_command=_http_step("detect"))
    cv_prepare = BashOperator(task_id="cv_prepare", bash_command=_http_step("prepare"))
    after_prepare = BranchPythonOperator(
        task_id="after_prepare",
        python_callable=_after_prepare_branch,
    )
    e2_done = EmptyOperator(task_id="e2_done", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    cv_train = BashOperator(task_id="cv_train", bash_command=_http_step("train"))
    cv_eval = BashOperator(task_id="cv_eval", bash_command=_http_step("eval"))
    cv_gate = BashOperator(task_id="cv_gate", bash_command=_http_step("gate"))

    mode_branch >> cv_split >> cv_detect >> cv_prepare
    mode_branch >> cv_prepare
    cv_prepare >> after_prepare
    after_prepare >> e2_done
    after_prepare >> cv_train >> cv_eval >> cv_gate
