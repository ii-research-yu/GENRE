FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# init
RUN apt-get update && apt-get upgrade -y && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -U pip

# torch
RUN pip install torchsummary
RUN pip install torch_optimizer

# pip
WORKDIR /tmp
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# KILT
WORKDIR /tmp
RUN git clone https://github.com/facebookresearch/KILT.git
WORKDIR /tmp/KILT
RUN python setup.py install

# fairseq
WORKDIR /tmp
RUN apt-get install -y build-essential
RUN git clone https://github.com/pytorch/fairseq
WORKDIR /tmp/fairseq
RUN pip install --editable ./

# GENRE
WORKDIR /tmp
RUN git clone https://github.com/facebookresearch/GENRE
WORKDIR /tmp/GENRE
RUN pip install -r requirements.txt
RUN pip install --editable ./
RUN pip install -U Cython h5py numpy

# clearn
RUN rm -rf /var/lib/apt/lists/*
