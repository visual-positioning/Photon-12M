# ---------- Base ----------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ---------- System deps ----------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libgl1 \
    openjdk-11-jre \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python ----------
RUN python3 -m pip install --upgrade pip

WORKDIR /workspace

# ---------- Requirements ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Project ----------
COPY . .

# ---------- Download MobileCLIP weights ----------
RUN mkdir -p models && \
    wget -O models/mobileclip_s1.pt \
    https://huggingface.co/apple/MobileCLIP-S1/resolve/main/mobileclip_s1.pt

# ---------- Default ----------
CMD ["bash"]
