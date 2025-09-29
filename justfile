# Justfile for cross-platform Docker builds

default: enable-emulation build-all

# Enable QEMU emulation
enable-emulation:
    docker run --privileged --rm tonistiigi/binfmt --install all

# build docker image for ARM32 (armv7l)
build-arm32-docker:
    docker buildx build --platform linux/arm/v7 \
        --cache-from type=gha,scope=arm32 \
        --cache-to type=gha,mode=max,scope=arm32 \
        --load \
        -t dockcross-uv-armv7l -f build/Dockerfile build

# Build for AARCH64
build-aarch64-docker:
    docker buildx build --platform linux/aarch64 \
        --cache-from type=gha,scope=aarch64 \
        --cache-to type=gha,mode=max,scope=aarch64 \
        --load \
        -t dockcross-uv-aarch64 -f build/Dockerfile build

build-arm32: build-arm32-docker
    docker run --platform linux/arm/v7 -v {{justfile_directory()}}:/work --rm dockcross-uv-armv7l /usr/bin/build.sh

# Run build container for AARCH64
build-aarch64: build-aarch64-docker
    docker run --platform linux/aarch64 -v {{justfile_directory()}}:/work --rm dockcross-uv-aarch64 /usr/bin/build.sh

build-all: build-arm32 build-aarch64

test-units:
    cd tests && uv run pytest .
