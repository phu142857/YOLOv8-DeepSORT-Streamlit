#!/bin/sh
set -eu

TOKEN_FILE="${STACK_AUTH_TOKEN_FILE:-/run/stack-auth/CV_MLAIR_TOKEN}"
if [ -f "$TOKEN_FILE" ]; then
  export CV_MLAIR_TOKEN="$(cat "$TOKEN_FILE")"
  export MLAIR_WORKER_TOKEN="${CV_MLAIR_TOKEN}"
  export ML_AIR_TRACKING_TOKEN="${CV_MLAIR_TOKEN}"
fi

exec "$@"
