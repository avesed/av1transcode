# The grain-auto service (av1transcode profile switch grain_auto): analysis, the v3 denoiser and AV1 grain synthesis on
# the B580. Built on the box, from images that only exist there:
#   av1t-cvvdp:gxx4   torch 2.14 + XPU, triton, ffmpeg with VA-API / QSV and the iHD driver (the CVVDP scorer's image)
#   grav1synth:local  grav1synth 0.2.0, linked against Ubuntu 22.04's ffmpeg 5 libraries, which come along below
# zcdec (the zero-copy reads' VA-API decoder, zc.py) is built here against a cut-down static ffmpeg of the engine's
# release, with the engine's hevc patch (zcdec/build.sh; the tarball comes from GitHub, so the build needs network).
# The weights are not in git: put v3g.pt, v3s2.pt, s32v2.pt, spynet.pt and v3g_mid_u8.pt2 (v3g's int8 middle:
# quantize_int8.py v3g.pt CAPTURES v3g_mid_u8.pt2 mid) into grain/weights/ first, then
#   docker build -t av1t-grain:latest grain/
FROM grav1synth:local AS g
RUN mkdir -p /out/lib && cp /usr/local/bin/grav1synth /out/ && \
    ldd /usr/local/bin/grav1synth | awk '/=> \// {print $3}' \
      | grep -vE '/(libc|libm|libgcc_s|libpthread|libdl|librt|libstdc\+\+)\.so|ld-linux' | xargs -I{} cp -L {} /out/lib/

FROM ubuntu:24.04 AS zcdec
ARG JOBS=8
COPY zcdec/ /build/zcdec/
RUN JOBS=$JOBS sh /build/zcdec/build.sh /build/zcdec

FROM av1t-cvvdp:gxx4
COPY --from=g /out /opt/grav1synth
COPY --from=zcdec /build/zcdec/zcdec /usr/local/bin/zcdec
RUN /opt/cvvdp/bin/pip install --no-cache-dir torchao==0.18.0      # the int8 middle (int8_net.py, quantize_int8.py)
RUN printf '#!/bin/sh\nLD_LIBRARY_PATH=/opt/grav1synth/lib exec /opt/grav1synth/grav1synth "$@"\n' > /usr/local/bin/grav1synth \
    && chmod +x /usr/local/bin/grav1synth
COPY . /grain
# zcshim: host code against the SYCL headers of torch's own runtime, so its Level Zero context is torch's
RUN cd /grain/zcdec && g++ -O2 -shared -fPIC -std=c++17 -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING -I/opt/cvvdp/include \
      zcshim.cpp -o libzcshim.so -L/opt/cvvdp/lib -lsycl -lze_loader && rm -f zcdec
ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages GRAIN_WEIGHTS=/grain/weights \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton
EXPOSE 8790
ENTRYPOINT ["/opt/cvvdp/bin/python", "/grain/server.py"]
