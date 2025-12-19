# Justfile for cross-platform Docker builds

default: enable-emulation build-all

# Enable QEMU emulation
enable-emulation:
    docker run --privileged --rm tonistiigi/binfmt --install all

# build docker image for ARM32 (armv7l) - Bookworm
build-arm32-docker:
    docker buildx build --platform linux/arm/v7 \
        --cache-from type=gha,scope=arm32 \
        --cache-to type=gha,mode=max,scope=arm32 \
        --load \
        -t dockcross-uv-armv7l -f build/Dockerfile build

# Build for AARCH64 - Bookworm
build-aarch64-docker:
    docker buildx build --platform linux/aarch64 \
        --cache-from type=gha,scope=aarch64 \
        --cache-to type=gha,mode=max,scope=aarch64 \
        --load \
        -t dockcross-uv-aarch64 -f build/Dockerfile build

# build docker image for ARM32 (armv7l) - Trixie
build-arm32-docker-trixie:
    docker buildx build --platform linux/arm/v7 \
        --cache-from type=gha,scope=arm32-trixie \
        --cache-to type=gha,mode=max,scope=arm32-trixie \
        --load \
        -t dockcross-uv-armv7l-trixie -f build/Dockerfile.trixie build

# Build for AARCH64 - Trixie
build-aarch64-docker-trixie:
    docker buildx build --platform linux/aarch64 \
        --cache-from type=gha,scope=aarch64-trixie \
        --cache-to type=gha,mode=max,scope=aarch64-trixie \
        --load \
        -t dockcross-uv-aarch64-trixie -f build/Dockerfile.trixie build

build-arm32: build-arm32-docker
    docker run --platform linux/arm/v7 -v {{justfile_directory()}}:/work --rm dockcross-uv-armv7l /usr/bin/build.sh bookworm

# Run build container for AARCH64
build-aarch64: build-aarch64-docker
    docker run --platform linux/aarch64 -v {{justfile_directory()}}:/work --rm dockcross-uv-aarch64 /usr/bin/build.sh bookworm

build-arm32-trixie: build-arm32-docker-trixie
    docker run --platform linux/arm/v7 -v {{justfile_directory()}}:/work --rm dockcross-uv-armv7l-trixie /usr/bin/build.sh trixie

# Run build container for AARCH64 - Trixie
build-aarch64-trixie: build-aarch64-docker-trixie
    docker run --platform linux/aarch64 -v {{justfile_directory()}}:/work --rm dockcross-uv-aarch64-trixie /usr/bin/build.sh trixie

build-all: build-arm32 build-aarch64

build-all-trixie: build-arm32-trixie build-aarch64-trixie

test-units:
    cd tests && uv run pytest .

format:
   cd rpi-vision-ai-processor/opt/rpivisionai/ && uv run --no-project black .

lint:
   uv run pylint rpi-vision-ai-processor/opt/rpivisionai/*.py