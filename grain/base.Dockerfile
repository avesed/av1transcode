# av1t-cvvdp: the grain service's base. PyTorch XPU in a venv (/opt/cvvdp) on top of the engine image, whose Level Zero /
# compute runtime / iHD stack is the one proven on the B580, plus what the grain service builds or compiles against.
# Every pip version is pinned by base-constraints.txt (production's freeze). Build on the box from grain/:
#   docker build -f base.Dockerfile --build-arg ENGINE=av1t:gpu-<commit> -t av1t-cvvdp:<commit> .
# It replaces av1t-cvvdp:gxx4 (av1t-cvvdp:test of history/cvvdp/Dockerfile plus four apt layers, FROM av1t:gpu-a10a4ac).
ARG ENGINE=ghcr.io/avesed/av1transcode:gpu
FROM ${ENGINE}
COPY base-constraints.txt /tmp/base-constraints.txt
RUN python3 -m venv /opt/cvvdp || (apt-get update && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/cvvdp)
# torch and torchvision from the same XPU index: a PyPI torchvision next to the XPU torch fails at import (operator
# torchvision::nms does not exist)
RUN /opt/cvvdp/bin/pip install --no-cache-dir -c /tmp/base-constraints.txt torch torchvision \
      --index-url https://download.pytorch.org/whl/xpu
RUN /opt/cvvdp/bin/pip install --no-cache-dir -c /tmp/base-constraints.txt cvvdp \
 && /opt/cvvdp/bin/python -c "import torch, torchvision, pycvvdp; print('torch', torch.__version__, 'tv', torchvision.__version__)"
# oneDNN (torch's XPU matmul/conv) JIT-compiles through OpenCL: without the ICD it finds the B580 on Level Zero and still
# fails with "OpenCL device not found". g++ / python3.12-dev: torch.compile's and Triton's host side; libze-dev and
# intel-ocloc: Triton's XPU backend (and zcshim's Level Zero header).
RUN apt-get update && apt-get install -y --no-install-recommends intel-opencl-icd ocl-icd-libopencl1 g++ python3.12-dev \
      libze-dev intel-ocloc && rm -rf /var/lib/apt/lists/*
ENTRYPOINT []
