# Justfile for cross-platform Docker builds

default: enable-emulation build-all

# Enable QEMU emulation
enable-emulation:
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# build docker image for ARM32 (armv7l)
build-arm32-docker:
    docker build --platform linux/arm -t dockcross-uv-armv7l -f build/Dockerfile.arm build

# Build for AARCH64
build-aarch64-docker:
    docker build --platform linux/aarch64 -t dockcross-uv-aarch64 -f build/Dockerfile.aarch64 build

build-arm32: build-arm32-docker
    docker run --platform linux/arm/v7 -v {{justfile_directory()}}:/work --rm -it dockcross-uv-armv7l /usr/bin/build.sh

# Run build container for AARCH64
build-aarch64: build-aarch64-docker
    docker run --platform linux/aarch64 -v {{justfile_directory()}}:/work --rm -it dockcross-uv-aarch64 /usr/bin/build.sh

build-all: build-arm32 build-aarch64