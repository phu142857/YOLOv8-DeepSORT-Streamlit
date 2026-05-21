#!/usr/bin/env sh
# MLAir API runs as appuser (uid 1000). Named volume ml_air_model_artifacts is often root-owned
# after first create → POST .../versions/import returns 500 and Hub shows no versions.
set -e
cid="${MLAIR_API_CONTAINER:-ml-air-api}"
docker exec -u root "$cid" sh -c '
  mkdir -p /mlair/artifacts/models /mlair/artifacts/datasets
  chown -R appuser:appuser /mlair/artifacts/models /mlair/artifacts/datasets
'
echo "Fixed permissions on /mlair/artifacts/{models,datasets} in $cid"
echo "Re-run sync (auto on cv-api start, or manual):"
echo "  curl -X POST 'http://localhost:8000/api/v1/registry/models/sync-full'"
echo "After 'docker compose down -v' if Hub empty but sync says already_aligned:"
echo "  curl -X POST 'http://localhost:8000/api/v1/registry/models/sync-full?force=1'"
echo "  # or: rm -f artifacts/.mlair_model_sync_state.json"
