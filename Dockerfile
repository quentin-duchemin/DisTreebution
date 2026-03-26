# Use the Dev Container base for a smooth VS Code experience
FROM mcr.microsoft.com/devcontainers/python:3.10

# 1. System-level dependencies
# We keep git for development and build tools for any C-extensions in your dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Handle specific version constraints and core scientific stack
# We combine these to reduce the number of image layers
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    "scikit-learn>=0.24.0" \
    quantile_forest \
    scipy \
    pandas \
    matplotlib \
    tqdm \
    ipykernel \
    jupyter \
    "nbformat>=5.0.0"
    
# 4. Set up the workspace
WORKDIR /workspaces/DisTreebution

# 5. Copy and install your package
# We copy everything first, then install in editable mode
COPY . .
RUN pip install --no-cache-dir -e .

