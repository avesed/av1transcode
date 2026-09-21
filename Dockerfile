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

# ---------- stage 3b: libvmaf (the VMAFx fork, for its SYCL backend) ----------
# Netflix's libvmaf has no Intel GPU path at all - its only accelerated backend
# is CUDA - so on this box (a B580, no CUDA) scoring ran entirely on the CPU,
# and scoring is the *larger* half of a probe: measured at the real probe scale
# (120 frames, native 4K) it is 5.6s on 10.7 cores and 7.6GB RSS against 3.0s
# on 4 cores for the preset-10 probe encode itself. Since _score_probe runs on
# every CRF bisection step of every shot, that is where the machine's memory
# and cores actually go.
#
# VMAFx is a fork that adds SYCL kernels while leaving the metric alone.
# Verified rather than assumed, 2026-09-01: 5 real sources (4K SDR/DV/HDR,
# 1080p SDR/HDR) x 3 CRFs, all scorers fed byte-identical y4m with the same
# model file. VMAFx on CPU and VMAFx on SYCL each agreed with the stock ffmpeg
# libvmaf to within 0.0001 VMAF on every one of the 15 points, and running
# those curves through the optimizer's own pick_crf over targets 88-96 moved
# the chosen CRF by at most 0.0006 - i.e. never. On the B580 the same probe
# scores in 1.7s on one core and 0.25GB.
#
# Pinned to a commit rather than a tag because the fork publishes no releases.
FROM ubuntu:24.04 AS vmaf-builder
ARG VMAFX_REF=0b58cb597680ef634c8cb15ef42c887e3465cfd0
ARG JOBS=8
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gpg gpg-agent software-properties-common \
        build-essential ninja-build nasm xxd pkg-config git python3 python3-pip \
    && curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor -o /usr/share/keyrings/oneapi.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneAPI.list \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update && apt-get install -y --no-install-recommends \
        intel-oneapi-compiler-dpcpp-cpp libze1 libze-dev \
    && rm -rf /var/lib/apt/lists/* \
    # noble ships meson 1.3.2; VMAFx's meson.build requires >= 1.4
    && pip3 install --break-system-packages --no-cache-dir 'meson>=1.4'
# libva, for libvmaf's VA-API surface import (vmaf_sycl_import_va_surface, see
# optimizer.vmaf_zero_copy). meson compiles the DMA-BUF path only when libva
# and libva-drm are found, and without them the symbol is still exported - as
# a stub returning -ENOTSUP, which is exactly what the previous image shipped.
# Its own layer so the 3.7GB oneAPI layer above stays cached.
RUN apt-get update && apt-get install -y --no-install-recommends libva-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/VMAFx/vmafx.git vmafx && \
    cd vmafx && git checkout -q ${VMAFX_REF}
# -Dsycl_icpx_aot_targets=bmg-g21: ahead-of-time compile for the B580 only.
# The default list carries 19 Intel GPU generations this fleet will never run.
# -Denable_tests=false is not tidiness - `meson install` otherwise relinks the
# test binaries, and those fail to link (unresolved vmaf_log under LTO), which
# takes the whole install step down with them.
# DNN/CUDA/HIP/Metal/MCP off: none of them are reachable here, and enable_dnn
# would drag in ONNX Runtime.
RUN . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && \
    cd /build/vmafx && \
    CC=icx CXX=icpx meson setup /bld core \
        --buildtype=release --prefix=/usr/local --libdir=lib \
        --default-library=shared \
        -Denable_sycl=true -Dsycl_icpx_aot_targets=bmg-g21 \
        -Denable_avx512=true -Denable_float=true \
        -Denable_tests=false -Denable_docs=false -Denable_tools=true \
        -Denable_dnn=disabled -Denable_cuda=false -Denable_hip=false \
        -Denable_metal=disabled -Denable_mcp=false -Denable_rust_features=false && \
    ninja -C /bld -j${JOBS} && \
    meson install -C /bld --destdir /out --no-rebuild && \
    mkdir -p /out/lib /out/include /out/lib/pkgconfig /out/share/model /out/bin /out/oneapi && \
    cp -a /out/usr/local/lib/libvmaf.so* /out/lib/ && \
    # the -ENOTSUP stub would pass every other check in this image
    grep -aq vaExportSurfaceHandle "$(readlink -f /out/lib/libvmaf.so)" && \
    cp -a /out/usr/local/lib/pkgconfig/*.pc /out/lib/pkgconfig/ && \
    cp -a /out/usr/local/include/libvmaf /out/include/ && \
    # all of them: the 4k model is what a 4K source should be scored with,
    # and neg is the variant that does not reward enhancement/sharpening
    cp -a /build/vmafx/model/*.json /out/share/model/ && \
    cp -a /out/usr/local/bin/vmaf /out/bin/ && \
    # The DPC++ runtime closure the built library actually needs: 11 files,
    # ~77MB, against 2.9GB for the whole oneAPI install. Derived by walking
    # ldd over the library, the UR adapter and libumf - not guessed. Note that
    # BOTH level_zero adapters are required: with only v1 present the UR loader
    # reports UR_RESULT_ERROR_UNSUPPORTED_VERSION and enumerates no device,
    # and libvmaf's CLI would then silently score on the CPU instead.
    O=$(ls -d /opt/intel/oneapi/compiler/*/lib | head -1) && \
    U=$(ls -d /opt/intel/oneapi/umf/*/lib | head -1) && \
    T=$(ls -d /opt/intel/oneapi/tcm/*/lib | head -1) && \
    for f in "$O"/libimf.so "$O"/libintlc.so.5 "$O"/libirc.so "$O"/libirng.so \
             "$O"/libsvml.so "$O"/libsycl.so.9 "$O"/libur_loader.so.0 \
             "$O"/libur_adapter_level_zero.so.0 "$O"/libur_adapter_level_zero_v2.so.0 \
             "$U"/libumf.so.1 "$T"/libhwloc.so.15 ; do \
        cp -aL "$f" /out/oneapi/ || exit 1 ; done && \
    test "$(ls /out/oneapi | wc -l)" = 11

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
    build-essential pkg-config curl nasm yasm ca-certificates patch \
    libvpx-dev libx264-dev libx265-dev libopus-dev libvorbis-dev \
    libmp3lame-dev libass-dev libfreetype-dev libfontconfig1-dev \
    libvulkan-dev liblcms2-dev libdav1d-dev \
    libva-dev libvpl-dev \
    libze-dev \
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
# libvmaf.so now has libsycl/libur/libumf in its DT_NEEDED, so configure's
# link test for vmaf_sycl_state_init cannot resolve without these present.
COPY --from=vmaf-builder /out/oneapi/ /usr/local/lib/
RUN ldconfig
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
# The release tarball from the GitHub mirror, not a clone of git.ffmpeg.org:
# that host answers ICMP but refuses TCP on 80/443/9418, which fails the build
# outright, and a tarball is 17MB against 137MB for the shallowest useful
# fetch. The tree carries a RELEASE file, so the version string stays correct
# without a .git directory.
# Vendored, not fetched: see patches/README.md, which also records why this is
# our patch and not the fork's (theirs needs two of the fork's earlier patches
# to place at all, and gates the code behind a CONFIG_ symbol it never
# defines). Adds sycl_device/sycl_profile to the existing libvmaf filter.
# The zerocopy patch adds the libvmaf_sycl filter, which takes VA-API frames;
# keep-decoder-until-cleanup stops ffmpeg destroying a decoder's VA context
# while its frames are still queued, which crashed that filter in iHD.
# keep-graph-on-equivalent-hwframes stops ffmpeg rebuilding a filter graph when
# a VA-API decoder swaps in an identical frames context at an in-band SPS
# change, which restarted setpts, -t, fps and libvmaf in the middle of a read.
# hevc-keep-hwaccel-on-equivalent-sps stops hevc destroying its VA context at
# all at such a change, when the new SPS keeps profile, size and format: that
# destroy raced the filter thread's syncs of queued frames and segfaulted iHD.
COPY patches/ffmpeg-n9.0.1-libvmaf-sycl.patch /build/
COPY patches/ffmpeg-n9.0.1-libvmaf-sycl-zerocopy.patch /build/
COPY patches/ffmpeg-n9.0.1-keep-decoder-until-cleanup.patch /build/
COPY patches/ffmpeg-n9.0.1-keep-graph-on-equivalent-hwframes.patch /build/
COPY patches/ffmpeg-n9.0.1-hevc-keep-hwaccel-on-equivalent-sps.patch /build/
RUN curl -fsSL "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/${FFMPEG_REF}.tar.gz" \
      | tar xz && \
    cd "FFmpeg-${FFMPEG_REF}" && \
    patch -p1 --fuzz=0 < /build/ffmpeg-n9.0.1-libvmaf-sycl.patch && \
    patch -p1 --fuzz=0 < /build/ffmpeg-n9.0.1-libvmaf-sycl-zerocopy.patch && \
    patch -p1 --fuzz=0 < /build/ffmpeg-n9.0.1-keep-decoder-until-cleanup.patch && \
    patch -p1 --fuzz=0 < /build/ffmpeg-n9.0.1-keep-graph-on-equivalent-hwframes.patch && \
    patch -p1 --fuzz=0 < /build/ffmpeg-n9.0.1-hevc-keep-hwaccel-on-equivalent-sps.patch && \
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
    # Prove the SYCL path is really compiled in, not just that the patch
    # applied: the option can exist while CONFIG_LIBVMAF_SYCL is 0, in which
    # case sycl_device=0 would fail at run time with ENOSYS instead of here.
    ffmpeg -hide_banner -h filter=libvmaf 2>&1 | grep -q sycl_device && \
    grep -q "^#define CONFIG_LIBVMAF_SYCL 1" config.h && \
    # component switches live in config_components.h, not config.h
    grep -q "^#define CONFIG_LIBVMAF_SYCL_FILTER 1" config_components.h && \
    ffmpeg -hide_banner -h filter=libvmaf_sycl 2>&1 | grep -q sycl_device && \
    # Both keep-graph hunks are in what was built, not only in the tree: the
    # fftools one in the binary, the hwdownload one in libavfilter (static, as
    # configure defaults to, so it is the archive the binary was linked from).
    grep -aq "Keeping filter graph" /usr/local/bin/ffmpeg && \
    grep -aq "Accepting frames from an equivalent hwframe context" /usr/local/lib/libavfilter.a && \
    # and the hevc keep-hwaccel hunk, in the static libavcodec the binary
    # was linked from
    grep -aq "Keeping hwaccel %s for a new SPS" /usr/local/lib/libavcodec.a && \
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
# Vision RPU on the GPU instead of on llvmpipe. libze1 + libze-intel-gpu1 are
# the Level Zero loader and driver, which is a separate stack from both VA-API
# and Vulkan and is what the SYCL libvmaf talks to.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopus0 libvpx9 libx264-164 libx265-199 libmp3lame0 libvorbis0a \
        libvorbisenc2 \
        libass9 libfreetype6 libfontconfig1 libvulkan1 \
        libgl1 libegl1 libopengl0 libdav1d7 mkvtoolnix \
        python3 python3-pip libpython3.12t64 \
        libzimg2 liblcms2-2 mesa-vulkan-drivers \
        libva2 libva-drm2 intel-media-va-driver libmfx-gen1 libvpl2 \
        libze1 libze-intel-gpu1 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*


# MediaInfo from MediaArea rather than Ubuntu's 24.01. Before 26.05 its AV1
# parser skips initial_display_delay, which SVT-AV1 v4 always writes, loses its
# place and drops the whole sequence header: every output then shows no Format
# profile, Bit depth or Chroma subsampling (fixed in MediaInfoLib bd835a3c08).
# Nothing in the app calls it - `cli check` only looks for it - so it is here
# for reading an output by hand: docker exec av1transcode mediainfo FILE.
# Pinned by version AND checksum, like the tessdata below; apt installs the
# three packages' own dependencies (libmms0, libtinyxml2, ...) from Ubuntu.
ARG MEDIAINFO_VERSION=26.05
ARG LIBZEN_VERSION=0.4.41
ARG MEDIAINFO_SHA256=00ef1b1b33f8b9cef72aebf912649273484c4708928d1aef6aab56f14c96dc1b
ARG LIBMEDIAINFO_SHA256=4a5d2ce1304b67f54bd85a89cdbd80d78aacef6e44bce6c0b25d2b9e150fe2f1
ARG LIBZEN_SHA256=56591441f8f475337ae3bbc2f8ceb76104ede84964fe9df8d0c1cc96460b37f0
RUN mkdir /tmp/mediainfo && cd /tmp/mediainfo \
    && base=https://mediaarea.net/download/binary \
    && curl -fsSLO "$base/libzen0/${LIBZEN_VERSION}/libzen0v5_${LIBZEN_VERSION}-1_amd64.Ubuntu_24.04.deb" \
    && curl -fsSLO "$base/libmediainfo0/${MEDIAINFO_VERSION}/libmediainfo0v5_${MEDIAINFO_VERSION}-1_amd64.Ubuntu_24.04.deb" \
    && curl -fsSLO "$base/mediainfo/${MEDIAINFO_VERSION}/mediainfo_${MEDIAINFO_VERSION}-1_amd64.Ubuntu_24.04.deb" \
    && printf '%s  %s\n' \
        "${LIBZEN_SHA256}" "libzen0v5_${LIBZEN_VERSION}-1_amd64.Ubuntu_24.04.deb" \
        "${LIBMEDIAINFO_SHA256}" "libmediainfo0v5_${MEDIAINFO_VERSION}-1_amd64.Ubuntu_24.04.deb" \
        "${MEDIAINFO_SHA256}" "mediainfo_${MEDIAINFO_VERSION}-1_amd64.Ubuntu_24.04.deb" \
        | sha256sum -c - \
    && apt-get update && apt-get install -y --no-install-recommends ./*.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/mediainfo \
    && mediainfo --Version | grep -q "v${MEDIAINFO_VERSION}"

# OCR for the English image subtitles (transcode.optimizer.pgs_ocr_srt, see
# app/pgsocr.py). tesseract-ocr is the engine; wamerican is the word list the
# out-of-vocabulary gate scores a track against, and without it that gate
# skips itself rather than passing everything.
#
# osd.traineddata is deleted in the same layer: it is the orientation and
# script detector, which is only ever loaded by --psm 0, and this code asks
# for --psm 6 and --psm 7. Measured, it is 10.3MB of the package.
#
# THE MODEL IS FETCHED, not the packaged one, and that is the one number worth
# arguing about. Ubuntu's tesseract-ocr-eng ships tessdata_fast (verified: its
# eng.traineddata is byte-identical to upstream tessdata_fast, md5
# d1be414fbb296b3ad777bfca655e194e). Measured on the same 711-cue track,
# scored against that file's own SDH text track:
#     packaged (fast)   0.3020% CER case-folded, 664/711 cues exact
#     tessdata_best     0.1276% CER case-folded, 689/711 cues exact
# So best halves the character error rate and turns 25 more cues perfect, for
# 11.3MB more model (15.4 against 4.1) and ~40% more OCR CPU (168.3 against
# 118.8 CPU-seconds for the whole track). A track is OCR'd once, forever, into
# a file people then read; the CPU is spent in the mux, which is minutes at
# the end of an encode that runs for hours. Deleting osd pays for most of the
# size difference on its own.
#
# Pinned by tag AND checksum: this is a 15MB binary blob fetched at build
# time, and a silent change in it would change every subtitle this ships.
ARG TESSDATA_BEST_REF=4.1.0
ARG TESSDATA_BEST_SHA256=8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr wamerican \
    && rm -f /usr/share/tesseract-ocr/*/tessdata/osd.traineddata \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o /tmp/eng.traineddata \
        "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/${TESSDATA_BEST_REF}/eng.traineddata" \
    && echo "${TESSDATA_BEST_SHA256}  /tmp/eng.traineddata" | sha256sum -c - \
    && mv /tmp/eng.traineddata \
        "$(dirname "$(find /usr/share/tesseract-ocr -name eng.traineddata | head -1)")/eng.traineddata" \
    && tesseract --list-langs 2>&1 | grep -qx eng \
    && test -s /usr/share/dict/words

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
COPY --from=vmaf-builder /out/oneapi/ /usr/local/lib/
# The libvmaf CLI, for diagnosing the GPU path by hand. Careful with it: unlike
# the filter, the CLI falls back to the CPU when SYCL init fails and still
# prints a valid score, so `--backend sycl` (not the default) is what actually
# proves a device is being used.
COPY --from=vmaf-builder /out/bin/vmaf /usr/local/bin/
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