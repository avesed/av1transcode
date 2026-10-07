# The grain-auto service (av1transcode profile switch grain_auto): analysis, the v3 denoiser and AV1 grain synthesis on
# the B580. Built on the box, from images that only exist there:
#   av1t-cvvdp:gxx4   torch 2.14 + XPU, triton, ffmpeg with VA-API / QSV and the iHD driver (the CVVDP scorer's image)
#   grav1synth:local  grav1synth 0.2.0, linked against Ubuntu 22.04's ffmpeg 5 libraries, which come along below
# The weights are not in git: put v3g.pt, v3s2.pt, s32v2.pt and spynet.pt into grain/weights/ first, then
#   docker build -t av1t-grain:latest grain/
FROM grav1synth:local AS g
RUN mkdir -p /out/lib && cp /usr/local/bin/grav1synth /out/ && \
    ldd /usr/local/bin/grav1synth | awk '/=> \// {print $3}' \
      | grep -vE '/(libc|libm|libgcc_s|libpthread|libdl|librt|libstdc\+\+)\.so|ld-linux' | xargs -I{} cp -L {} /out/lib/

FROM av1t-cvvdp:gxx4
COPY --from=g /out /opt/grav1synth
RUN printf '#!/bin/sh\nLD_LIBRARY_PATH=/opt/grav1synth/lib exec /opt/grav1synth/grav1synth "$@"\n' > /usr/local/bin/grav1synth \
    && chmod +x /usr/local/bin/grav1synth
COPY . /grain
ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages GRAIN_WEIGHTS=/grain/weights \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton
EXPOSE 8790
ENTRYPOINT ["/opt/cvvdp/bin/python", "/grain/server.py"]
