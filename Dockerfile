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

# PyTorch: default CPU wheels; PYTORCH_WHEEL=cu124 for local NVIDIA GPU train.
ARG PYTORCH_WHEEL=cpu
RUN pip install --upgrade pip wheel \
    && pip install 'setuptools>=69.0.0,<81' \
    && if [ "$PYTORCH_WHEEL" = "cpu" ]; then \
         pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
       else \
         pip install torch torchvision --index-url https://download.pytorch.org/whl/${PYTORCH_WHEEL}; \
       fi \
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
