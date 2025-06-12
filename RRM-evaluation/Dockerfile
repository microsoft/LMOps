# Use public Nvidia images (rather than Beaker), for reproducibility
FROM --platform=linux/amd64 nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 
#nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt update && apt install -y openjdk-8-jre-headless

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

# Install base tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    jq \
    language-pack-en \
    make \
    sudo \
    unzip \
    vim \
    wget \
    parallel \
    iputils-ping \
    tmux

# This ensures the dynamic linker (or NVIDIA's container runtime, I'm not sure)
# puts the right NVIDIA things in the right place (that THOR requires).
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Install conda. We give anyone in the users group the ability to run
# conda commands and install packages in the base (default) environment.
# Things installed into the default environment won't persist, but we prefer
# convenience in this case and try to make sure the user is aware of this
# with a message that's printed when the session starts.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh \
    && echo "32d73e1bc33fda089d7cd9ef4c1be542616bd8e437d1f77afeeaf7afdb019787 Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" \
        | sha256sum --check \
    && bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

ENV PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Ensure users can modify their container environment.
RUN echo '%users ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

WORKDIR /stage/
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN pip install --upgrade pip setuptools wheel
# designed for cuda 12.1
RUN pip3 install torch torchvision torchaudio
# If you need to use cuda 11.8, use this and the below vllm code for installing with cuda 11.8
# RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# Install vLLM with CUDA 11.8.
# RUN export VLLM_VERSION=0.6.1.post1
# RUN export PYTHON_VERSION=310
# RUN pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

COPY rewardbench rewardbench
COPY scripts scripts
COPY setup.py setup.py
COPY Makefile Makefile
COPY README.md README.md
RUN pip install -e .[generative]
RUN chmod +x scripts/*

# this is just very slow
RUN pip install flash-attn==2.6.3 --no-build-isolation

# for olmo-instruct v1, weird install requirements
# RUN pip install ai2-olmo 

# for better-pairRM
RUN pip install jinja2 

# generative installs
RUN pip install anthropic
RUN pip install openai
RUN pip install together
RUN pip install google-generativeai

# for interactive session
RUN chmod -R 777 /stage/
