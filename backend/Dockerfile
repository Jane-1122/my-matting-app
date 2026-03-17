FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
# 免费档内存有限，禁用预加载确保服务能启动；首次预览会较慢
ENV DISABLE_PRELOAD=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY app/ app/
RUN mkdir -p models data/uploads data/results data/temp \
    && wget -q -O models/u2net.onnx "https://huggingface.co/tomjackson2023/rembg/resolve/main/u2net.onnx?download=true" \
    && wget -q -O models/isnet-general-use.onnx "https://huggingface.co/x-Liola-x/isnet-general-use-onnx/resolve/main/isnet-general-use.onnx?download=true"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
