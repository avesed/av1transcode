# syntax=docker/dockerfile:1.4
# =====================================================================
# AV1 Transcode Archive - all-in-one image
# Native ffmpeg 9.0.1 + SVT-AV1 v4.2 + av1an + dovi_tool +
# mkvtoolnix + mediainfo + Python scheduler/UI.
#
#   docker build -t av1transcode .
#   docker compose up -d
# =====================================================================

# ---------- stage 1: SVT-AV1 v4.2 ----------
FROM debian:bookworm-slim AS svt-builder
ARG SVT_AV1_TAG=v4.2.0
ARG JOBS=8
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config git cmake ninja-build nasm ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${SVT_AV1_TAG} \
        https://gitlab.com/AOMediaCodec/SVT-AV1.git svt-av1 && \
    cd svt-av1 && mkdir build && cd build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
          -DBUILD_TESTING=OFF .. && ninja -j${JOBS} && ninja install
# copy install artifacts (prefix /usr/local)
RUN mkdir -p /out/lib /out/bin /out/include /out/lib/pkgconfig && \
    cp -a /usr/local/lib/libSvtAv1* /out/lib/ 2>/dev/null; \
    cp -a /usr/local/lib/pkgconfig/SvtAv1Enc.pc /out/lib/pkgconfig/ 2>/dev/null; \
    cp /usr/local/bin/SvtAv1EncApp /out/bin/ 2>/dev/null; \
    cp -r /usr/local/include/svt-av1 /out/include/ 2>/dev/null; true

# ---------- stage 2: libplacebo v7 (ffmpeg master needs >= 7.351.0) ----------
FROM debian:bookworm-slim AS libplacebo-builder
ARG PLACEBO_TAG=v7.360.1
ARG JOBS=8
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential meson ninja-build pkg-config git ca-certificates \
    libvulkan-dev libgl-dev libegl-dev libgl1-mesa-dev \
    liblcms2-dev glslang-dev \
    && rm -rf /var/lib/apt/lists/*
ENV C_INCLUDE_PATH=/build/libplacebo/3rdparty/Vulkan-Headers/include
RUN git clone --recursive --depth 1 --branch ${PLACEBO_TAG} \
        https://github.com/haasn/libplacebo.git libplacebo && \
    cd libplacebo && \
    git submodule update --init --recursive && \
    meson setup build --buildtype=release \
        -Dvulkan=enabled -Dglslang=enabled -Dshaderc=disabled \
        -Dopengl=disabled -Dd3d11=disabled -Dlcms=enabled -Ddovi=enabled \
        -Ddemos=false \
        --prefix=/usr/local --libdir=lib && \
    ninja -C build -j${JOBS} && \
    meson install -C build --destdir /out && \
    mkdir -p /out/lib /out/include /out/lib/pkgconfig && \
    cp -a /out/usr/local/lib/*.so* /out/lib/ && \
    cp -a /out/usr/local/lib/pkgconfig/*.pc /out/lib/pkgconfig/ && \
    cp -a /out/usr/local/include/* /out/include/

# ---------- stage 3: Vulkan headers bumped for ffmpeg master (needs >= 1.3.277) ----------
FROM debian:bookworm-slim AS vulkan-headers-builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates && rm -rf /var/lib/apt/lists/* && \
    git clone --depth 1 --branch vulkan-sdk-1.4.350.1 \
        https://github.com/KhronosGroup/Vulkan-Headers.git vulkan-headers && \
    mkdir -p /out/include && \
    cp -a vulkan-headers/include/vulkan /out/include/ && \
    cp -a vulkan-headers/include/vk_video /out/include/ && \
    printf 'prefix=/usr\nincludedir=${prefix}/include\nName: vulkan\nDescription: Vulkan loader\nVersion: 1.4.350\nCflags: -I${includedir}\nLibs: -lvulkan\n' > /out/vulkan.pc

# ---------- stage 3b: libvmaf v2.3.1 (not packaged in bookworm) ----------
FROM debian:bookworm-slim AS vmaf-builder
ARG VMAF_TAG=v2.3.1
ARG JOBS=8
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential meson ninja-build nasm xxd pkg-config git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${VMAF_TAG} \
        https://github.com/Netflix/vmaf.git vmaf && \
    cd vmaf/libvmaf && \
    meson setup build --buildtype=release --prefix=/usr/local --libdir=lib \
        -Denable_avx512=true && \
    ninja -C build -j${JOBS} && \
    meson install -C build --destdir /out && \
    mkdir -p /out/lib /out/include /out/lib/pkgconfig /out/share/model && \
    cp -a /out/usr/local/lib/*.so* /out/lib/ && \
    cp -a /out/usr/local/lib/pkgconfig/*.pc /out/lib/pkgconfig/ && \
    cp -a /out/usr/local/include/libvmaf /out/include/ && \
    cp -a /build/vmaf/model/vmaf_v0.6.1.json /out/share/model/ && \
    cp -a /out/usr/local/bin/vmaf /out/bin/ 2>/dev/null; true

# ---------- stage 4: ffmpeg 9.0.1 release (DV support) ----------
FROM debian:bookworm-slim AS ffmpeg-builder
# A release tag, and it is actually used - this ARG was previously declared and
# then ignored while the source was cloned from master, so no two builds of this
# image contained the same ffmpeg. The optimizer's measurement path depends on
# specific libvmaf, libplacebo and framesync behaviour, which is not something
# to re-roll on every rebuild. Verified present in 9.0.1: framesync
# ts_sync_mode, libplacebo apply_dolbyvision, the dovi_rpu/dovi_split bitstream
# filters, and libsvtav1's SVT_AV1_CHECK_VERSION(4,0,0) path for the SVT-AV1
# v4.2.0 built above. Bump deliberately.
ARG FFMPEG_REF=n9.0.1
ARG JOBS=8
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config curl nasm yasm ca-certificates \
    libvpx-dev libx264-dev libx265-dev libopus-dev libvorbis-dev \
    libmp3lame-dev libass-dev libfreetype-dev libfontconfig1-dev \
    libvulkan-dev liblcms2-dev libdav1d-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=svt-builder /out/lib/ /usr/local/lib/
COPY --from=svt-builder /out/include/ /usr/local/include/
COPY --from=libplacebo-builder /out/lib/ /usr/local/lib/
COPY --from=libplacebo-builder /out/include/ /usr/local/include/
COPY --from=vulkan-headers-builder /out/include/vulkan /usr/include/vulkan/
COPY --from=vulkan-headers-builder /out/include/vk_video /usr/include/vk_video/
COPY --from=vulkan-headers-builder /out/vulkan.pc /usr/local/lib/pkgconfig/vulkan.pc
COPY --from=vmaf-builder /out/lib/ /usr/local/lib/
COPY --from=vmaf-builder /out/include/ /usr/local/include/
COPY --from=vmaf-builder /out/lib/pkgconfig/ /usr/local/lib/pkgconfig/
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
# The release tarball from the GitHub mirror, not a clone of git.ffmpeg.org:
# that host answers ICMP but refuses TCP on 80/443/9418, which fails the build
# outright, and a tarball is 17MB against 137MB for the shallowest useful
# fetch. The tree carries a RELEASE file, so the version string stays correct
# without a .git directory.
RUN curl -fsSL "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/${FFMPEG_REF}.tar.gz" \
      | tar xz && \
    cd "FFmpeg-${FFMPEG_REF}" && \
    ./configure --prefix=/usr/local \
        --enable-gpl --enable-nonfree \
        --enable-libvpx --enable-libx264 --enable-libx265 \
        --enable-libopus --enable-libvorbis --enable-libmp3lame \
        --enable-libsvtav1 --enable-libdav1d --enable-libvmaf \
        --enable-libplacebo --enable-vulkan \
        --enable-libass --enable-libfreetype --enable-libfontconfig \
        --disable-doc --disable-debug && \
    make -j${JOBS} && make install && \
    mkdir -p /out/lib /out/bin && \
    cp -a /usr/local/lib/* /out/lib/ && \
    cp /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /out/bin/

# ---------- stage 4a: zimg >=3.0.5 (bookworm ships 3.0.4; VS R73 needs 3.0.5) ----------
FROM debian:bookworm-slim AS zimg-builder
ARG ZIMG_TAG=release-3.0.5
ARG JOBS=8
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential autoconf automake libtool pkg-config git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${ZIMG_TAG} \
        https://github.com/sekrit-twc/zimg.git zimg && \
    cd zimg && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local && \
    make -j${JOBS} && \
    make install DESTDIR=/out && \
    mkdir -p /out/lib /out/include /out/lib/pkgconfig && \
    cp -a /out/usr/local/lib/*.so* /out/lib/ && \
    cp -a /out/usr/local/lib/pkgconfig/*.pc /out/lib/pkgconfig/ && \
    cp -a /out/usr/local/include/* /out/include/

# ---------- stage 4b: VapourSynth (av1an hard-links it) ----------
FROM debian:bookworm-slim AS vapoursynth-builder
ARG VS_VERSION=R73
ARG JOBS=8
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential meson ninja-build nasm pkg-config git ca-certificates \
    python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages -q meson cython
COPY --from=zimg-builder /out/lib/ /usr/local/lib/
COPY --from=zimg-builder /out/include/ /usr/local/include/
COPY --from=zimg-builder /out/lib/pkgconfig/ /usr/local/lib/pkgconfig/
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
RUN git clone --depth 1 --branch ${VS_VERSION} \
        https://github.com/vapoursynth/vapoursynth.git vapoursynth && \
    cd vapoursynth && \
    meson setup build --buildtype=release --prefix=/usr/local --libdir=lib \
        -Denable_python_module=true -Denable_vspipe=false \
        -Denable_vsscript=true -Denable_x86_asm=true \
        -Dpython3_bin=/usr/bin/python3 && \
    ninja -C build -j${JOBS} && \
    meson install -C build --destdir /out && \
    mkdir -p /out/lib /out/include /out/lib/pkgconfig && \
    cp -a /out/usr/local/lib/*.so* /out/lib/ && \
    cp -a /out/usr/local/lib/pkgconfig/*.pc /out/lib/pkgconfig/ && \
    cp -a /out/usr/local/include/vapoursynth* /out/include/ && \
    find /out/usr/local -name "vapoursynth*.so" -exec cp -a {} /out/lib/ \; && \
    mkdir -p /out/python && \
    find /out/usr/local -type d -name "site-packages" -exec cp -a {}/vapoursynth* /out/python/ \; 2>/dev/null; \
    find /out -path "*site-packages*" -name "vapoursynth*" -exec cp -a {} /out/python/ \; 2>/dev/null; \
    cp -a /out/lib/vapoursynth.cpython-311-x86_64-linux-gnu.so /out/python/vapoursynth.cpython-311-x86_64-linux-gnu.so; true

# ---------- stage 5: av1an ----------
FROM rust:1-bookworm AS av1an-builder
RUN apt-get update && apt-get install -y --no-install-recommends nasm pkg-config \
        && rm -rf /var/lib/apt/lists/*
COPY --from=vapoursynth-builder /out/lib/ /usr/local/lib/
COPY --from=vapoursynth-builder /out/include/ /usr/local/include/
COPY --from=zimg-builder /out/lib/ /usr/local/lib/
ENV VAPOURSYNTH_LIB_DIR=/usr/local/lib
RUN cargo install av1an --locked --root /usr/local

# ---------- stage 6: dovi_tool ----------
FROM rust:1-bookworm AS dovi-builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake pkg-config \
    && rm -rf /var/lib/apt/lists/* && \
    git clone --depth 1 https://github.com/quietvoid/dovi_tool.git /build && \
    cd /build && cargo build --release && \
    cp target/release/dovi_tool /usr/local/bin/
# Runtime image (all-in-one)
# =====================================================================
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libopus0 libvpx7 libx264-164 libx265-199 libmp3lame0 libvorbis0a \
        libvorbisenc2 \
        libass9 libfreetype6 libfontconfig1 libvulkan1 \
        libgl1 libegl1 libopengl0 libdav1d6 mediainfo mkvtoolnix \
        python3 python3-pip libpython3.11 \
        libzimg2 liblcms2-2 mesa-vulkan-drivers \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg-builder /out/bin/ffmpeg /out/bin/ffprobe /usr/local/bin/
RUN mv /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg-real
COPY ffmpeg-wrap.py /usr/local/bin/ffmpeg
RUN chmod +x /usr/local/bin/ffmpeg
COPY --from=ffmpeg-builder /out/lib/ /usr/local/lib/
COPY --from=svt-builder /out/bin/SvtAv1EncApp /usr/local/bin/
COPY --from=libplacebo-builder /out/lib/ /usr/local/lib/
COPY --from=libplacebo-builder /out/include/ /usr/local/include/
COPY --from=vapoursynth-builder /out/lib/ /usr/local/lib/
COPY --from=zimg-builder /out/lib/ /usr/local/lib/
COPY --from=vmaf-builder /out/lib/ /usr/local/lib/
COPY --from=vmaf-builder /out/share/model/ /usr/share/model/
COPY --from=av1an-builder /usr/local/bin/av1an /usr/local/bin/
COPY --from=dovi-builder /usr/local/bin/dovi_tool /usr/local/bin/
COPY --from=vapoursynth-builder /out/python/ /usr/local/lib/python3.11/dist-packages/
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH=/usr/local/lib/python3.11/dist-packages

# Python scheduler + UI
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt
COPY app /app/app
COPY config.yaml /app/config.yaml
WORKDIR /app
RUN mkdir -p /media/input /media/output /media/rpu /media/work /data/logs

EXPOSE 8080
ENTRYPOINT ["python3", "-m", "app.cli"]
CMD ["run"]