# Use an official PyTorch image
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependancies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    ninja-build \
    libgl1 \
    cmake \
    gdb \
    lldb \
    valgrind \
    iputils-ping \
    libeigen3-dev \
    ffmpeg\
    libsm6 \
    libxext6 \
    libopenblas-dev \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Install IPOPT
# RUN apt-get update && apt-get -y install coinor-libipopt-dev

# Install PyQt5 Dependencies
# RUN apt-get update && apt-get install -y \
#     '^libxcb.*-dev' \
#     libx11-xcb-dev \
#     libglu1-mesa-dev \
#     libxrender-dev \
#     libxi-dev \
#     libxkbcommon-dev \
#     libxkbcommon-x11-dev

# utilities
RUN apt-get update && apt-get install -y tmux
    
# Set up user inside container to avoid issues with permissions.
ARG USERNAME=cho
ARG USER_UID=1094
ARG USER_GID=1094

RUN groupadd --gid $USER_GID $USERNAME \
&& useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
&& chown -R $USERNAME /opt/conda

# give user sudo privilege inside container
RUN usermod -aG sudo $USERNAME

USER $USERNAME
WORKDIR /workspace
