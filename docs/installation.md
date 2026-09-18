# Installation

## Docker (Recommended)

The easiest way to reproduce the environment is using the provided Dockerfile.

### Build the Image

```bash
docker build -t mobilecap .
```
This automatically downloads the required MobileCLIP-S1 weights into the `models/` directory during build.

### Run the Container

```bash
docker run --gpus all -it -v $(pwd):/workspace mobilecap
```
Make sure you have the `nvidia-container-toolkit` installed.

## Native Installation

### Requirements
- Python 3
- PyTorch 2.0+ and Torchvision 0.15+
- CUDA 12.1+ (Tested on CUDA 12.8 in paper, Docker uses 12.1)
- Java 11 (Required for METEOR/SPICE evaluation)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download MobileCLIP Weights
You must manually download the MobileCLIP-S1 weights if not using Docker:
```bash
mkdir -p models
wget -O models/mobileclip_s1.pt https://huggingface.co/apple/MobileCLIP-S1/resolve/main/mobileclip_s1.pt
```
