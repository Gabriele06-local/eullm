# EuLLM Forge — the verticalization pipeline, PyTorch on NVIDIA GPUs.
#
# Built from the repository root, like the engine: the pipeline's last stage
# turns the model into a GGUF with llama.cpp's converter and llama-quantize,
# and the image takes both from the engine's pinned llama.cpp submodule, so
# the GGUF Forge writes is one this repository's engine was tested to read.
#
#   git submodule update --init --recursive
#   docker build -f forge/Dockerfile -t eullm/forge .
#   docker compose --profile forge build forge
#
# Run it with the NVIDIA Container Toolkit on the host:
#
#   docker run --gpus all --shm-size=8g -v forge-output:/output eullm/forge \
#       forge Qwen/Qwen3-14B --profile legal-it
#
# PyTorch's wheels carry their own CUDA libraries: the host needs only a
# driver new enough for the wheel's CUDA (cu130: r580 or newer). For an older
# driver: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126

ARG UBUNTU_VERSION=24.04

# ── Stage 1: llama-quantize ─────────────────────────────────────
# Only the quantizer, CPU code: it runs once per export and needs no GPU.
# GGML_NATIVE off, so the binary does not depend on the CPU it was built on.
FROM ubuntu:${UBUNTU_VERSION} AS llama-tools

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/ llama.cpp/
RUN cmake -S llama.cpp -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF \
        -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_APP=OFF \
    && cmake --build build --target llama-quantize -j"$(nproc)"

# ── Stage 2: Runtime ────────────────────────────────────────────
# Plain Ubuntu, not a CUDA image: what PyTorch needs of CUDA comes with its
# wheels, and the driver with the NVIDIA Container Toolkit.
FROM ubuntu:${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: llama-quantize runs on OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# A virtual environment, not the system Python: Ubuntu 24.04 refuses pip
# installs into its own (PEP 668), and a venv keeps Forge's packages apart
# from the distribution's.
RUN python3 -m venv /opt/forge
ENV VIRTUAL_ENV=/opt/forge \
    PATH=/opt/forge/bin:$PATH

# PyTorch first, from the index of the CUDA it is built for: from PyPI, pip
# would pick whatever CUDA build PyPI carries. The cache mount keeps the
# wheels (several GB with CUDA) from one build to the next.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url "${TORCH_INDEX}"

# Forge itself, and what llama.cpp's converter needs beyond it (its tokenizer
# formats: sentencepiece, and the protobuf version that reads them).
WORKDIR /opt/eullm
COPY forge/pyproject.toml forge/
COPY forge/eullm_forge/ forge/eullm_forge/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ./forge \
        "sentencepiece>=0.1.98,<0.3.0" \
        "protobuf>=4.21.0,<5.0.0"

# The converter and the gguf package it reads from beside it, as the
# submodule pins them, and the quantizer built above. LLAMA_CPP_PATH is where
# Forge's export stage looks for them.
ARG LLAMA_CPP=engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp
COPY ${LLAMA_CPP}/convert_hf_to_gguf.py /opt/llama.cpp/
COPY ${LLAMA_CPP}/conversion/ /opt/llama.cpp/conversion/
COPY ${LLAMA_CPP}/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=llama-tools /src/build/bin/llama-quantize /opt/llama.cpp/build/bin/llama-quantize
ENV LLAMA_CPP_PATH=/opt/llama.cpp

# The same user, by number, as the engine and the hub: the compose file
# shares the `models` volume between them, and whichever container creates
# it first decides who owns it.
RUN groupadd -r -g 10001 eullm \
    && useradd -r -u 10001 -g eullm -m -d /home/eullm -s /usr/sbin/nologin eullm \
    && mkdir -p /models /output /data/huggingface \
    && chown eullm:eullm /models /output /data /data/huggingface
USER eullm
WORKDIR /output

VOLUME ["/models", "/output", "/data"]
ENV HF_HOME=/data/huggingface

ENTRYPOINT ["eullm-forge"]
CMD ["--help"]
