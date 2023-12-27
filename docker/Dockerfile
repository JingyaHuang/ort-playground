FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
CMD nvidia-smi

# Ignore interactive questions during `docker build`
ENV DEBIAN_FRONTEND noninteractive

# Versions
# available options 3.8, 3.9, 3.10, 3.11
ARG PYTHON_VERSION=3.9
ARG TORCH_CUDA_VERSION=cu118
ARG TORCH_VERSION=2.0.0
ARG TORCHVISION_VERSION=0.15.1

# Bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Install and update tools to minimize security vulnerabilities
RUN apt-get update
RUN apt-get install -y software-properties-common wget apt-utils patchelf git libprotobuf-dev protobuf-compiler cmake \
    bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 mercurial subversion libopenmpi-dev ffmpeg && \
    apt-get clean
RUN unattended-upgrade
RUN apt-get autoremove -y

# Install miniconda (comes with python 3.9 default)
ARG BUILD_USER=onnxruntimedev
ARG MINICONDA_PREFIX=/home/$BUILD_USER/miniconda3
RUN apt-get install curl

ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh
RUN curl -fSsL --insecure ${CONDA_URL} -o install-conda.sh && \
    /bin/bash ./install-conda.sh -b -p $MINICONDA_PREFIX && \
    $MINICONDA_PREFIX/bin/conda clean -ya && \
    $MINICONDA_PREFIX/bin/conda install -y python=${PYTHON_VERSION}

ENV PATH=$MINICONDA_PREFIX/bin:${PATH}

ARG PYTHON_EXE=$MINICONDA_PREFIX/bin/python

RUN $PYTHON_EXE -m pip install transformers datasets accelerate evaluate coloredlogs absl-py rouge_score seqeval scipy sacrebleu nltk scikit-learn parameterized sentencepiece sentence_transformers

# PyTorch
RUN $PYTHON_EXE -m pip install onnx ninja
RUN $PYTHON_EXE -m pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} -f https://download.pytorch.org/whl/${TORCH_CUDA_VERSION}

WORKDIR .

CMD ["/bin/bash"]

