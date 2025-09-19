# Dockerfile for dockcross + uv
FROM debian:bookworm

# Install dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git python3-dev python3-numpy python3-opencv build-essential pkg-config

# Download the latest uv installer
RUN curl -LsSf https://astral.sh/uv/install.sh > /uv-installer.sh \
  # Run the installer then remove it
  && UV_INSTALL_DIR="/usr/bin" sh /uv-installer.sh && rm /uv-installer.sh

COPY build.sh /usr/bin
# Optionally set a default working directory
WORKDIR /work