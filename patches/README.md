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
