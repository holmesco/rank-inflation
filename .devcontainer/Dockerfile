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
    libboost-all-dev \
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

# more utilities
RUN apt-get update && apt-get install -y tmux

# Install cuHALLaR and add to path
ADD cuHALLaR.tar.xz /opt
RUN chmod +x /opt/cuHALLaR/bin/cuHallar
ENV PATH="/opt/cuHALLaR/bin:${PATH}"

# Set locale (required for cuHALLaR)
# Install the locales package
RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/*
# Select and generate the specific locale (e.g., en_US.UTF-8)
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
# Set the environment variables
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install original clipper
RUN mkdir extern/clipper/build \
    && cd extern/clipper/build \
    && cmake .. \
    && make \
    && make pip-install 

# Install libraries for SuiteSparse
RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*
# Install SuiteSparse from scratch (assumed already cloned to extern/SuiteSparse)
RUN cd extern/SuiteSparse \
    && mkdir -p build && cd build \
    && cmake .. \
    && cmake --build . \
    && cmake --install .



# Set up user inside container to avoid issues with permissions.
ARG USERNAME=cho
ARG USER_UID=1094
ARG USER_GID=1094

RUN groupadd --gid $USER_GID $USERNAME \
&& useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
&& chown -R $USERNAME /opt/conda

RUN chown -R $USERNAME /opt/cuHALLaR

USER $USERNAME
WORKDIR /workspace
