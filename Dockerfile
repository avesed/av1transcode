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
FROM ubuntu:24.04 AS svt-builder
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
FROM ubuntu:24.04 AS libplacebo-builder
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
FROM ubuntu:24.04 AS vulkan-headers-builder
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
FROM ubuntu:24.04 AS vmaf-builder
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
    # all of them: the 4k model is what a 4K source should be scored with,
    # and neg is the variant that does not reward enhancement/sharpening
    cp -a /build/vmaf/model/*.json /out/share/model/ && \
    cp -a /out/usr/local/bin/vmaf /out/bin/ 2>/dev/null; true

# ---------- stage 4: ffmpeg 9.0.1 release (DV support) ----------
FROM ubuntu:24.04 AS ffmpeg-builder
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
# Intel's Arc PPA. Battlemage (B580, PCI 0xe20b) needs a media stack newer than
# any distro ships: Ubuntu's own intel-media-va-driver 24.1.0 fails
# vaInitialize on it, and Debian cannot get there at all - which is why this
# image is Ubuntu rather than bookworm. Verified against the card itself:
# VA-API 1.24 with VAProfileHEVCMain10 decode, and Mesa 25.2.8 reporting
# "Intel(R) Arc(tm) B580 Graphics (BMG G21)" where bookworm's Mesa 22.3 warned
# "Driver does not support the 0xe20b PCI ID" and fell back to llvmpipe - which
# is what the Dolby Vision libplacebo path had been running on all along.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent ca-certificates \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get purge -y software-properties-common gpg-agent \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config curl nasm yasm ca-certificates \
    libvpx-dev libx264-dev libx265-dev libopus-dev libvorbis-dev \
    libmp3lame-dev libass-dev libfreetype-dev libfontconfig1-dev \
    libvulkan-dev liblcms2-dev libdav1d-dev \
    libva-dev libvpl-dev \
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
        --enable-vaapi --enable-libvpl \
        --enable-libass --enable-libfreetype --enable-libfontconfig \
        --disable-doc --disable-debug && \
    make -j${JOBS} && make install && \
    mkdir -p /out/lib /out/bin && \
    cp -a /usr/local/lib/* /out/lib/ && \
    cp /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /out/bin/

# ---------- stage 4a: zimg >=3.0.5 (bookworm ships 3.0.4; VS R73 needs 3.0.5) ----------
FROM ubuntu:24.04 AS zimg-builder
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
FROM ubuntu:24.04 AS vapoursynth-builder
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
    cp -a /out/lib/vapoursynth.cpython-312-x86_64-linux-gnu.so /out/python/vapoursynth.cpython-312-x86_64-linux-gnu.so; true

# ---------- stage 4c: VapourSynth plugins for target_metric=ssimulacra2 ----------
# Prebuilt wheels rather than source builds: vszip is written in Zig (which
# would mean shipping that toolchain) and bestsource's README only claims
# FFmpeg 8.x support against the 9.0.1 built above. Both wheels carry a plain
# plugin .so plus bundled deps, so nothing here links against our ffmpeg or
# VapourSynth - verified loading into VapourSynth R73, which exposes
# vszip.SSIMULACRA2(reference, distorted).
FROM ubuntu:24.04 AS vsplugin-fetcher
ARG VSZIP_VERSION=22.1.0
ARG BESTSOURCE_VERSION=21.0
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN set -eu; \
    base=https://files.pythonhosted.org; \
    for pkg in "vapoursynth-vszip:${VSZIP_VERSION}:manylinux_2_17_x86_64" \
               "vapoursynth-bestsource:${BESTSOURCE_VERSION}:manylinux_2_28_x86_64"; do \
      name=${pkg%%:*}; rest=${pkg#*:}; ver=${rest%%:*}; plat=${rest#*:}; \
      url=$(curl -fsSL "https://pypi.org/pypi/${name}/${ver}/json" \
            | python3 -c "import json,sys;print(next(f['url'] for f in json.load(sys.stdin)['urls'] if '${plat}' in f['filename']))"); \
      curl -fsSL -o "${name}.whl" "$url"; \
      python3 -c "import zipfile;zipfile.ZipFile('${name}.whl').extractall('x')"; \
    done; \
    mkdir -p /out/vapoursynth; \
    find x -name "*.so" -o -name "*.so.*" | while read -r f; do cp -a "$f" /out/vapoursynth/; done; \
    ls -1 /out/vapoursynth/
# vszip ships avx2/znver4 variants alongside the baseline; keep the baseline
# under the plain name so it loads on any x86-64.
RUN cd /out/vapoursynth && rm -f libvszip.zn4.so && \
    if [ -f libvszip.avx2.so ]; then rm -f libvszip.avx2.so; fi && \
    test -f libvszip.so && test -f libbestsource.so

# ---------- stage 5: av1an ----------
# Ubuntu + rustup rather than the rust:1-bookworm image: av1an links against
# the VapourSynth built above, and that is now a noble build. Linking a
# bookworm binary (glibc 2.36) against a noble .so (which needs 2.38+) fails
# outright - the compatibility only runs the other way.
FROM ubuntu:24.04 AS av1an-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential nasm pkg-config curl ca-certificates git \
        && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH
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
FROM ubuntu:24.04 AS runtime

# Intel's Arc PPA - see the note in the ffmpeg builder for why this image is
# Ubuntu and not Debian.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent ca-certificates \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get purge -y software-properties-common gpg-agent \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# libvpx9 / libdav1d7 / libpython3.12t64 are the noble spellings of what
# bookworm called libvpx7 / libdav1d6 / libpython3.11.
#
# The Intel block is what lets this image drive the B580. intel-media-va-driver
# is the VA-API driver, libmfx-gen1 the QSV runtime behind libvpl, and
# mesa-vulkan-drivers (25.2.8 here) is what libplacebo needs to apply a Dolby
# Vision RPU on the GPU instead of on llvmpipe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopus0 libvpx9 libx264-164 libx265-199 libmp3lame0 libvorbis0a \
        libvorbisenc2 \
        libass9 libfreetype6 libfontconfig1 libvulkan1 \
        libgl1 libegl1 libopengl0 libdav1d7 mediainfo mkvtoolnix \
        python3 python3-pip libpython3.12t64 \
        libzimg2 liblcms2-2 mesa-vulkan-drivers \
        libva2 libva-drm2 intel-media-va-driver libmfx-gen1 libvpl2 \
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
COPY --from=vapoursynth-builder /out/python/ /usr/local/lib/python3.12/dist-packages/
# VapourSynth plugins backing target_metric=ssimulacra2 (see stage 4c)
COPY --from=vsplugin-fetcher /out/vapoursynth/ /usr/local/lib/vapoursynth/
# The plugin dir is on the library path too: the bestsource wheel carries
# its own ffmpeg/lcms/dav1d and has no RPATH, so the loader has to be told
# where those sit or the plugin fails to load.
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/vapoursynth
ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages

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