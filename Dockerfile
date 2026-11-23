# CPU image (runs anywhere). For GPUs use a CUDA-enabled base, e.g.
#   docker build --build-arg BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime -t mlperf-gpu .
#   docker run --gpus all mlperf-gpu run --config configs/gpu.yaml
ARG BASE=python:3.11-slim
FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends g++ make && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# torch is already present in the pytorch/* base images; only install it for the plain-python base
RUN python -c "import torch" 2>/dev/null || pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    pip install --no-cache-dir pyyaml pytest
COPY . .
RUN make -C cpp ceilings_cpu
ENTRYPOINT ["python", "-m", "mlperf"]
CMD ["run", "--config", "configs/cpu_smoke.yaml"]
