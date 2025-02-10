FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
# RUN pip install tqdm
# RUN pip install gguf
# RUN pip install accelerate

# Install system dependencies
RUN apt update && apt install -y \
    python3 python3-pip python3-venv git \
    build-essential cmake ninja-build && \
    ln -s /usr/bin/python3 /usr/bin/python

# Install llama-cpp-python
RUN pip install --no-cache-dir llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# Ensure CUDA libraries are optional at runtime
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && ldconfig || true

CMD ["/bin/bash"]
