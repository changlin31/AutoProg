FROM nvcr.io/nvidia/pytorch:20.11-py3

WORKDIR /code

RUN pip install -U timm==0.4.5
RUN pip install -U tlt==0.1.0

COPY . .
