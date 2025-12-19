#!/bin/bash

set -e

# Get distribution parameter (default to bookworm for backward compatibility)
DISTRO=${1:-bookworm}

export UV_LINK_MODE=copy
cp -r rpi-vision-ai-processor/ /tmp
BUILD_PATH="/tmp/rpi-vision-ai-processor/opt/rpivisionai"
VENV="$BUILD_PATH/.venv"
if [ -d "$VENV" ]
then
    rm -r "$VENV"
fi

# 1. Create a clean Python virtualenv using `uv` with system python
uv venv --system-site-packages --python python3 "$VENV"
source $VENV/bin/activate

cd $BUILD_PATH
# 2. Install Python package and dependencies
uv pip install --index-strategy unsafe-best-match -r pyproject.toml

ARCH=$(dpkg --print-architecture)
sed -i "s/^Architecture: .*/Architecture: $ARCH/" /tmp/rpi-vision-ai-processor/DEBIAN/control

mkdir /tmp/dist

# 3. Build the .deb
dpkg-deb --build /tmp/rpi-vision-ai-processor /tmp/dist

# 4. Rename the package to include distribution
cd /tmp/dist
for deb in *.deb; do
    # Extract package name, version, and arch from the original filename
    # Format: packagename_version_arch.deb
    if [[ $deb =~ ^(.+)_([0-9.]+)_([^_]+)\.deb$ ]]; then
        PACKAGE_NAME="${BASH_REMATCH[1]}"
        VERSION="${BASH_REMATCH[2]}"
        PACKAGE_ARCH="${BASH_REMATCH[3]}"
        NEW_NAME="${PACKAGE_NAME}_${VERSION}-${DISTRO}_${PACKAGE_ARCH}.deb"
        mv "$deb" "$NEW_NAME"
    fi
done

cp /tmp/dist/*.deb /work