FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git \
    build-essential cmake ninja-build \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install llama-cpp-python with cuda support
RUN pip install --no-cache-dir llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 --target=/opt/llama-cpp-cuda

# Install llama-cpp without cuda
RUN pip install --no-cache-dir llama-cpp-python

# Install trie library
RUN pip install marisa-trie

# Ensure CUDA is optional at runtime
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && ldconfig || true

CMD ["/bin/bash"]
