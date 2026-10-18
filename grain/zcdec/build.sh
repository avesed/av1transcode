#!/bin/sh
# zcdec against a static ffmpeg cut down to what it needs: the demuxers grain sources come in, the decoders VA-API
# can take, libdrm for the DRM PRIME mapping. Same release and same hevc patch as the engine's ffmpeg (the decoder
# keeps its VA context across an equivalent in-band SPS). Run in ubuntu:24.04 with the zcdec/ directory at $1.
set -eu
D=${1:-/build/zcdec}; REF=${FFMPEG_REF:-n9.0.1}; JOBS=${JOBS:-8}
apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config curl nasm ca-certificates \
    patch libva-dev libdrm-dev && rm -rf /var/lib/apt/lists/*
cd /tmp && curl -fsSL "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/$REF.tar.gz" | tar xz && cd "FFmpeg-$REF"
patch -p1 --fuzz=0 < "$D/ffmpeg-n9.0.1-hevc-keep-hwaccel-on-equivalent-sps.patch"
./configure --prefix=/opt/zcff --disable-programs --disable-doc --disable-debug --disable-everything \
    --enable-static --disable-shared --enable-vaapi --enable-libdrm --enable-protocol=file \
    --enable-demuxer=matroska,mov,ivf,mpegts,mpegps,hevc,h264,obu \
    --enable-parser=hevc,h264,av1,mpegvideo,vc1,vp9 \
    --enable-decoder=hevc,h264,av1,mpeg2video,vc1,vp9 \
    --enable-hwaccel=hevc_vaapi,h264_vaapi,av1_vaapi,mpeg2_vaapi,vc1_vaapi,vp9_vaapi
grep -q "^#define CONFIG_VAAPI 1" config.h && grep -q "^#define CONFIG_LIBDRM 1" config.h
make -j"$JOBS" >/dev/null && make install >/dev/null
gcc -O2 -Wall -o "$D/zcdec" "$D/zcdec.c" -I/opt/zcff/include -L/opt/zcff/lib -lavformat -lavcodec -lavutil \
    -lva -lva-drm -ldrm -lm -lpthread
"$D/zcdec" 2>&1 | head -1 || true
