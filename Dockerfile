FROM python:3.13-slim
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
# RUN pip install tqdm
# RUN pip install gguf
# RUN pip install accelerate

# RUN apt-get update && apt-get install -y python3 python3-pip

# Install system dependencies
RUN apt update && apt install -y \
    build-essential cmake ninja-build git

# Install llama-cpp-python
RUN pip install --no-cache-dir llama-cpp-python --force-reinstall --upgrade

CMD ["/bin/bash"]
