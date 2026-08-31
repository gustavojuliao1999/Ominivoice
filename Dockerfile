FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# HuggingFace cache — override com -e HF_HOME=/runpod-volume/huggingface no RunPod
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY server.py .
COPY handler.py .

EXPOSE 8000

# RunPod Serverless entry point.
# Para rodar a API FastAPI localmente, sobrescreva:
#   docker run --gpus all -p 8000:8000 ominivoice uvicorn server:app --host 0.0.0.0 --port 8000
#CMD ["python", "server.py"]
CMD ["python", "handler.py"]
