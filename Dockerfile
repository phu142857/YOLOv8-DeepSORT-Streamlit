# CV Lifecycle Workload — CPU inference image (mount weights/ at runtime)
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

# PyTorch CPU first, then app deps. Skip `pip install -e .` — setup.py needs pkg_resources
# in an isolated PEP517 env; PYTHONPATH=/app is enough for `import ultralytics`.
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY shared ./shared
COPY inference ./inference
COPY workers ./workers
COPY mlair_adapter ./mlair_adapter
COPY examples ./examples
COPY scripts ./scripts
COPY .streamlit ./.streamlit

RUN mkdir -p /app/artifacts /app/weights/detection

EXPOSE 8000 8501
