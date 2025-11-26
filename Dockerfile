FROM nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        ffmpeg \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project
COPY . /workspace

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Default command for RunPod Serverless
CMD ["python3", "-u", "handler.py"]


