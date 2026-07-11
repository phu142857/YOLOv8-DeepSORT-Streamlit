# YOLO → MLAir: resource usage via the official SDK (in-worker)

Resource sampling for CV tasks uses MLAir's **official SDK** `sdk.resource_monitor`
running **inside the worker**, not a bespoke YOLO monitor.

## Why in-worker (not server-side)

Tasks run in **external execution mode** — the CV worker is a separate container. MLAir's
worker task API (`worker_tasks.py`) only *stores* usage the worker posts
(`heartbeat.usage`, `complete.usage_samples`/`resource_usage`); there is **no** server-side
sampler that can see into the worker container. If the worker sends nothing, MLAir records
only `runtime_seconds` (from lease→complete timestamps) and every CPU/RAM/GPU field stays
`null` with `sample_count=0`. So the SDK monitor must run in the worker; MLAir aggregates
what it sends. (A brief experiment removing worker-side sampling produced exactly the
empty-usage symptom and was reverted.)

## What YOLO wires

| Area | Detail |
|------|--------|
| Sampling engine | `sdk.resource_monitor.ResourceMonitor` (autostarts on current pid) |
| Heartbeat | `worker_task_runtime._heartbeat_loop` posts `usage=monitor.latest_heartbeat_usage()` (~3s) to renew the lease |
| Complete/fail POST | `sdk.worker_client.post_task_complete_from_bundle(usage_bundle=monitor.complete_bundle())` / `post_task_fail(usage_bundle=...)` |
| Worker runtime | `worker_task_runtime.run_handler_with_heartbeat` → returns `(plugin_result, complete_bundle)` |
| Result mapping | `run_tracking_client.{metrics,artifacts}_from_plugin_result` |

## SDK delivery — vendored `mlair` wheel (network-only architecture)

MLAir runs purely as the prebuilt image `ml-air:latest` (like an external MLflow server);
this repo never references the ml-air source tree. The SDK is installed into the CV image
from a **vendored wheel** `vendor/mlair-*.whl` (the wheel exported by ml-air's own
packaging, which bundles both `mlair` and `sdk`):

```dockerfile
COPY vendor ./vendor
RUN pip install ./vendor/mlair-*.whl
```

`import sdk` / `import mlair` then resolve from the installed package (no bind-mount, no
`../ml-air` path). To refresh the wheel when the SDK changes, rebuild it from ml-air once
and drop the new `.whl` into `vendor/`.

## Server-side plugin contracts — derived image `ml-air-cv:latest`

MLAir loads plugins only via Python entry points (`group="mlair.plugins"`) and validates
pipelines server-side, so the `cv_yolo_*` contracts must live inside the MLAir API process.
`ml-air:latest` ships pure (only built-in plugins), so `compose.yaml`'s `mlair` service is a
thin derived image `ml-air-cv:latest` (`Dockerfile.mlair-cv`): `FROM ml-air:latest` +
`pip install integrations/mlair_cv_plugins`. That package is **contract-only**
(name/version/engine_version/inputs/outputs/lineage/ui_schema + `validate()`) with zero deps
— no torch/ultralytics; the real training runs in the external CV worker. This layers only
YOLO's own package onto the prebuilt image and never references or rebuilds the ml-air source.
`GET /v1/plugins` then lists `cv_yolo_split/detect/prepare/train/eval/gate` +
`cv_hard_example_mine` with no errors.

## SDK accuracy caveats (fix in ml-air `sdk/resource_monitor.py`)

The SDK monitor works, but under rootless podman GPU-in-container is weaker than the old
YOLO adapter: NVML reports host-namespace PIDs so **per-process VRAM may not attribute**
(reads low/0), and CPU% can read 0 on the first sample of a fresh `psutil.Process`.
CPU/RAM/duration are fine. These are SDK-level issues to patch upstream (host↔container PID
mapping, `psutil.Process` reuse, device-level VRAM fallback), then re-export the vendored wheel.
