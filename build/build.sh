#!/bin/bash

set -e

export UV_LINK_MODE=copy
cp -r rpi-vision-ai-processor/ /tmp 
BUILD_PATH="/tmp/rpi-vision-ai-processor/opt/rpivisionai"
VENV="$BUILD_PATH/.venv"
if [ -d "$VENV" ]
then
    rm -r "$VENV"
fi

# 1. Create a clean Python virtualenv using `uv`
uv venv --system-site-packages "$VENV"
source $VENV/bin/activate

cd $BUILD_PATH
# 2. Install Python package and dependencies
uv pip install --index-strategy unsafe-best-match -r pyproject.toml

ARCH=$(dpkg --print-architecture)
sed -i "s/^Architecture: .*/Architecture: $ARCH/" /tmp/rpi-vision-ai-processor/DEBIAN/control

mkdir /tmp/dist

# 3. Build the .deb
dpkg-deb --build /tmp/rpi-vision-ai-processor /tmp/dist
cp /tmp/dist/*.deb /work