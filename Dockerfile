# CV Lifecycle Workload — CPU/GPU inference + MLAir external worker.
# MLAir SDK wheel is built from the ml-air:latest image (compose service mlair-base).

ARG MLAIR_BASE_IMAGE=ml-air:latest

FROM ${MLAIR_BASE_IMAGE} AS mlair-src

FROM python:3.11-slim-bookworm AS mlair-wheel
RUN pip install --no-cache-dir build wheel setuptools
WORKDIR /src
COPY --from=mlair-src /app/sdk ./sdk
COPY --from=mlair-src /app/mlair ./mlair
RUN cat > pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mlair"
version = "0.1.1"
requires-python = ">=3.11"
dependencies = ["PyYAML>=6.0.1"]

[tool.setuptools.packages.find]
where = ["."]
include = ["mlair*", "sdk*"]
EOF
RUN pip wheel . -w /wheels --no-deps

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
COPY ultralytics ./ultralytics
COPY config.py utils.py app.py ./

ARG PYTORCH_WHEEL=cpu
ARG MLAIR_BASE_IMAGE=ml-air:latest
COPY --from=mlair-wheel /wheels /wheels
RUN pip install --upgrade pip wheel \
    && pip install 'setuptools>=69.0.0,<81' \
    && if [ "$PYTORCH_WHEEL" = "cpu" ]; then \
         pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
       else \
         pip install torch torchvision --index-url https://download.pytorch.org/whl/${PYTORCH_WHEEL}; \
       fi \
    && pip install -r requirements.txt \
    && pip install --force-reinstall --no-deps /wheels/mlair-*.whl

COPY backend ./backend
COPY frontend ./frontend
COPY shared ./shared
COPY inference ./inference
COPY workers ./workers
COPY mlair_adapter ./mlair_adapter
COPY examples ./examples
COPY scripts ./scripts
COPY .streamlit ./.streamlit

RUN chmod +x /app/scripts/docker-entrypoint-cv.sh \
    && mkdir -p /app/artifacts /app/weights/detection

EXPOSE 8000 8501
ENTRYPOINT ["/app/scripts/docker-entrypoint-cv.sh"]
