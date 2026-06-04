#!/usr/bin/env python3
"""HTTP sidecar for Airflow baseline — one lifecycle step per POST /step."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlair_adapter.torch_compat import apply_torch_checkpoint_compat

apply_torch_checkpoint_compat()

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "airflow_mlair_baseline_step",
    ROOT / "scripts" / "airflow_mlair_baseline_step.py",
)
_step_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader
_spec.loader.exec_module(_step_mod)
STEP_HANDLERS = _step_mod.STEP_HANDLERS
_build_context = _step_mod._build_context
_should_skip_step = _step_mod._should_skip_step


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/step":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        step = str(body.get("step") or "")
        if step not in STEP_HANDLERS:
            self.send_error(400, f"unknown step {step}")
            return
        for key, val in (body.get("env") or {}).items():
            if val is not None:
                os.environ[str(key)] = str(val)
        import argparse

        ns = argparse.Namespace(
            step=step,
            mode=body.get("mode"),
            dataset_version_id=body.get("dataset_version_id"),
            train_ready_version_id=body.get("train_ready_version_id"),
            model_id=body.get("model_id"),
            baseline_run_id=body.get("baseline_run_id"),
        )
        if _should_skip_step(step):
            out = {"ok": True, "skipped": True, "step": step}
        else:
            ctx = _build_context(ns)
            out = STEP_HANDLERS[step](ctx)
        payload = json.dumps(out, default=str).encode("utf-8")
        self.send_response(200 if out.get("ok") else 500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        print(fmt % args, flush=True)


def main() -> None:
    port = int(os.getenv("BASELINE_RUNNER_PORT", "9191"))
    print(f"baseline-cv-runner listening on :{port}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
