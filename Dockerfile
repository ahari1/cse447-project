FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
# RUN pip install tqdm
RUN pip install transformers
RUN pip install torch
