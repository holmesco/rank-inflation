# A Repository for Exploring SDP Solution Rank Changing Tools

# Requirements

See `.devcontainer/Dockerfile` for required tools.

# Python install instructions (uv)

Install system dependencies (needed for `scikit-sparse`):

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libsuitesparse-dev libopenblas-dev gfortran
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

Create the environment and install all project dependencies (including CUDA 12.1 PyTorch wheels):

```bash
cd /workspace
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

Run tests:

```bash
pytest
```

# Legacy pip install instructions

Run 
```bash
pip install -v .
```