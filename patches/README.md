# patches/

Vendored patches applied during the Docker build. Kept in-tree rather than
fetched at build time so a rebuild does not depend on a third-party host
staying up, and so the diff is reviewable here.

## `ffmpeg-n9.0.1-libvmaf-sycl.patch`

Adds two options to FFmpeg's existing `libvmaf` filter, against the exact tag
`FFMPEG_REF` pins:

- `sycl_device` (int, default **-1 = off**) - score on this SYCL device
- `sycl_profile` (bool, default 0) - profile the SYCL queue

When `sycl_device >= 0` the filter calls `vmaf_sycl_state_init()` +
`vmaf_sycl_import_state()`, and **returns `AVERROR(EINVAL)` if either fails**.
That is deliberate. libvmaf's own `vmaf` CLI does the opposite: on a failed
SYCL init it falls back to the CPU and returns a perfectly valid score, so a
broken driver would surface only as everything being three times slower and
using thirty times the cores. Here it either runs on the GPU or stops.
`app/optimizer.py` pays one throwaway comparison per job up front so that a
missing device costs one clear warning instead of one failure per probe.

### Why this is ours and not the fork's

The VMAFx fork ships its own `ffmpeg-patches/0003-libvmaf-wire-sycl-backend-
selector.patch`, and using it directly does not work:

- Its hunks are cut against a tree that already has the fork's patches `0001`
  and `0002` applied. Against pristine n9.0.1, two of five hunks need `--fuzz=3`
  to place at all, and what they then produce does not compile
  (`vf_libvmaf.c:98: error: expected expression before '{' token`).
- More fundamentally, it tests `#if CONFIG_LIBVMAF_SYCL`, and nothing in that
  patch ever defines it - the symbol arrives later in the fork's series. So the
  SYCL code compiled out even where the patch did apply.

This patch does the same job in a way that stands alone: it registers
`libvmaf_sycl` in `CONFIG_EXTRA` (which is what actually emits
`CONFIG_LIBVMAF_SYCL`) and adds the `check_pkg_config` probe next to the
existing `libvmaf_cuda` one, mirroring how FFmpeg already handles the CUDA
backend. It applies to n9.0.1 at `--fuzz=0`. Taking the fork's other patches
was never wanted anyway - `0001`/`0002` pull in tiny models and a DNN filter
that need ONNX Runtime, which the libvmaf build deliberately disables.

## `ffmpeg-n9.0.1-libvmaf-sycl-zerocopy.patch`

Adds a `libvmaf_sycl` filter that scores **VA-API frames without downloading
them** (`optimizer.vmaf_zero_copy`). Each input's surface (`data[3]`) and the
display of that frame's own device go to libvmaf's
`vmaf_sycl_import_va_surface`, which exports the surface as a DMA-BUF, imports
it through Level Zero and de-tiles the luma plane on the GPU.

Ported from the fork's `ffmpeg-patches/0005`, which needs the fork's
`0001`-`0004` to apply, takes only QSV frames with one display for both inputs,
and on a failed import **skips the frame** and scores the rest. A score pooled
over fewer frames than were compared is a wrong score, so here a failed import
fails the run. Applies on top of `ffmpeg-n9.0.1-libvmaf-sycl.patch` at
`--fuzz=0`; it needs libvmaf built with libva (see the Dockerfile).

## `ffmpeg-n9.0.1-keep-decoder-until-cleanup.patch`

One line out of `fftools/ffmpeg_dec.c`: the decoder thread no longer frees its
codec context as it exits. `dec_free()` frees it at cleanup anyway, after the
filtergraphs.

Freeing it early destroys a hwaccel's VA-API decode context while frames it
decoded are still queued in the filtergraph, and iHD's `vaSyncSurface` on one
of those surfaces dereferences the destroyed context. With `libvmaf_sycl` that
was a segfault in `iHD_drv_video.so` on the filter thread in ~9% of 4K probe
scores (gdb: `vmaf_sycl_import_va_surface -> vaSyncSurface` on `fc0` while
`decoder_thread -> avcodec_free_context -> ff_vaapi_decode_uninit ->
vaDestroyContext` ran on the AV1 decoder thread), each one resetting the
B580's compute engine. With the patch, 247 zero-copy scores ran with no
segfault and no engine reset. Any filter-thread sync of VA-API frames after
their decoder exits (hwdownload, hwmap) is exposed to the same race.
