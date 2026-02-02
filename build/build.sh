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

# 3. Download and install go2rtc binary
echo "Downloading go2rtc for architecture: $ARCH"
GO2RTC_VERSION="latest"
GO2RTC_URL="https://github.com/AlexxIT/go2rtc/releases/latest/download"

# Determine go2rtc binary based on architecture
case "$ARCH" in
    arm64|aarch64)
        GO2RTC_BINARY="go2rtc_linux_arm64"
        ;;
    armhf|armv7l)
        GO2RTC_BINARY="go2rtc_linux_arm"
        ;;
    *)
        echo "Warning: Unsupported architecture $ARCH for go2rtc, skipping download"
        GO2RTC_BINARY=""
        ;;
esac

if [ -n "$GO2RTC_BINARY" ]; then
    echo "Downloading $GO2RTC_BINARY..."
    mkdir -p /tmp/rpi-vision-ai-processor/usr/local/bin
    curl -L -o /tmp/rpi-vision-ai-processor/usr/local/bin/go2rtc \
        "${GO2RTC_URL}/${GO2RTC_BINARY}"
    chmod +x /tmp/rpi-vision-ai-processor/usr/local/bin/go2rtc
    echo "go2rtc downloaded and installed"
else
    echo "Skipping go2rtc installation for unsupported architecture"
fi

mkdir /tmp/dist

# 4. Build the .deb
dpkg-deb --build /tmp/rpi-vision-ai-processor /tmp/dist

# 5. Rename the package to include distribution
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