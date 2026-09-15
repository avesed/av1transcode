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

## `ffmpeg-n9.0.1-keep-graph-on-equivalent-hwframes.patch`

Stops ffmpeg rebuilding a filter graph when a VA-API decoder hands it frames
from a new but identical frames context. hevc runs `get_format()` again at
every in-band SPS change, and that always allocates a new `hw_frames_ctx`,
even when nothing about the surfaces changed. `fftools` compared the contexts
by pointer, logged "Reconfiguring filter graph because hwaccel changed" and
rebuilt the graph in the middle of the read, so every stateful filter in it
started over:

- `setpts=PTS-STARTPTS` rebased to 0 again
- the input `-t` (a trim in the graph) counted its duration again
- `fps` dropped the frame it was holding
- `libvmaf`/`libvmaf_sycl` wrote its log for the first segment, and a second
  instance overwrote it with the rest

On Stranger Things S05E02 (an SPS/PPS change at 11.053s) that failed
`av1_vaapi` card probes with `AVERROR_BUG` or made a 64-frame window come out
as 77 frames, and a zero-copy score kept 59 of its 84 frames. A software
decode of the same file never rebuilt.

A new context on the same device with the same format, `sw_format`, width and
height now keeps the graph (logged at `-v verbose` as "Keeping filter graph").
`hwdownload`, which demanded the configured context by pointer and would
otherwise fail the read with `EINVAL`, accepts the same equivalence and says
so once ("Accepting frames from an equivalent hwframe context"). The build
greps the `ffmpeg` binary and `libavfilter.a` for those two strings, so a
re-cut that loses either hunk fails the image build instead of a read.

What it does not cover - the patch header has the source citations for each:

- **Real changes still rebuild** and log the same INFO line: a new coded size
  or bit depth, another device, a switch between hardware and software frames.
  A new display size (conformance window), colour description or range
  rebuilds **software decodes too**, because fftools compares those on the
  frame itself, so rereading that window in software does not fix it.
- **QSV is left alone.** Its VPP binds a session to the input context's surface
  array. `hevc_qsv` graphs, such as the scenedetect pass, still rebuild at a
  decoder reinit.
- **VRAM.** The old decoder pool stays allocated until the command ends: one
  extra pool per change a read crosses, estimated at about 0.23-0.58 GiB for
  4K P010 (about 0.12-0.3 GiB for 8-bit NV12). Not measured.
- **iHD.** Frames of the old context still inside a kept graph are now synced
  after the decoder destroyed that context, where a rebuild used to free them.
  iHD's source makes that safe only once `vaDestroyContext` has returned and
  while the surface is still in that context's 127-entry render-target table.
  The first does not hold for the up to 2 frames already queued at the
  change: they can be synced while the decoder thread is still destroying
  the context, and this patch does not change that. That race has been seen
  in production (iHD segfaults on `fc0`, with compute and blitter engine
  resets, in zero-copy scores crossing the S05E03 and S05E04 closing SPS
  changes). `ffmpeg-n9.0.1-hevc-keep-hwaccel-on-equivalent-sps.patch` removes
  it for SPS changes the hwaccel can decode as they are, by not destroying
  the context there.

So `app/optimizer.py` still watches its reads for the rebuild line. A rebuilt
VA-API read of a window falls back to software for that window. A rebuild on
a software read cannot be fixed by rereading, so it logs one warning for that
window and the job carries on.

Verified offline only: it applies at `--fuzz=0` to n9.0.1 on its own and after
the three patches above, and both changed files compile without warnings.

## `ffmpeg-n9.0.1-hevc-keep-hwaccel-on-equivalent-sps.patch`

Stops hevc tearing down and rebuilding its VA-API hwaccel at an in-band SPS
change that does not affect it. hevc runs `get_format()` at every SPS
activation, and `ff_get_format()` starts by destroying the VA decode context,
config and frames context, even when the new SPS differs only in HRD values
or transform depths. The Stranger Things remuxes do exactly that at the logo
-> programme IRAP and again before the closing.

That destroy is what crashed production. The decoder thread destroys the
context while the filter thread is still syncing frames it already queued, and
nothing in iHD serialises the two: `vaDestroyContext` tears down the codec
pipeline (`codecHal->Destroy()`, `MOS_Delete`, then `pDecCtx` cleared) while
`vaSyncSurface`'s `StatusCheck`/`StatusReport` on the other thread is calling
into it. Zero-copy scores crossing the S05E03 (3975.305s) and S05E04
(5016.511s) closing changes segfaulted iHD on `fc0` at `+0x90a61c`
("segfault at 10") with compute and blitter engine resets. From the faulting
instructions and the 26.3.2 headers that is
`adapter->m_decoder->m_statusReport->m_completedCount` with `m_statusReport`
already deleted - inferred from offsets, not symbolised. The earlier
`+0x9a9d5c` crash (same iHD build) starts with the same instructions: the AV1
adapter's copy of the same function, when its decoder was freed at thread exit
(see keep-decoder above). A NULL check on `pDecCtx` in iHD would not fix it.

Now, before the old SPS is dropped, hevc checks whether the running hwaccel
can decode the new SPS as it is. If so it skips `ff_get_format()`: VA config,
VA context, frames context and private data stay as they were, and one INFO
line is logged, "Keeping hwaccel hevc_vaapi for a new SPS: coded WxH, FMT,
profile P unchanged (packet pts N)". Reads run at `-loglevel info`, so that
line is the production signal that a splice was kept. Kept only when all of
these hold:

- the hwaccel is VA-API and initialised, and there is an old SPS;
- same coded size, and the frames context has it;
- same `sps->pix_fmt`, equal to `sw_pix_fmt` (bit depth and chroma format are
  fixed in iHD for the life of a context). Only an 8-bit 4:2:0 range change
  (`yuv420p` <-> `yuvj420p`) re-initialises; a 10-bit range change keeps the
  hwaccel, which is safe because VA-API takes no range;
- same `general_profile_idc`, and it is Main, Main 10 or Main Still Picture
  (RExt and SCC always re-initialise);
- same VPS object, which hevc keeps only for a byte-identical repeat.

Measured on the production files (CPU only), all of these hold at the splices
that matter (S05E02 start 11.053s, S05E03 and S05E04 ends): the keyframe is
`IDR_N_LP` with in-band VPS/SPS/PPS and no RASL pictures, the VPS is
byte-identical, and both SPS variants are Main 10, 3840x2160, 10-bit 4:2:0.

With it, `keep-graph-on-equivalent-hwframes` is no longer reached at these
changes (the frames keep the same `hw_frames_ctx` pointer, so no second pool
is allocated either). It stays for the changes this patch still re-initialises
and for other decoders. `keep-decoder-until-cleanup` is still needed for the
one destroy left per decoder. The build greps `libavcodec.a` for "Keeping
hwaccel %s for a new SPS", so a re-cut that loses the hunk fails the image
build.

What it does not cover - the patch header has the source citations:

- **Real changes still re-initialise**, and keep the destroy-vs-sync race: a
  new coded size, bit depth, chroma format, 8-bit range, profile or VPS, and
  any RExt/SCC stream. Such a read still logs "Reconfiguring filter graph",
  which `app/optimizer.py` keeps watching for.
- **DPB size is not compared.** No pool is sized from it under libva >= 1 and
  iHD only dumps it. The VPS carries its own DPB size, so this only matters
  when the VPS is byte-identical and only the SPS values change.
- **Other hwaccels and `hevc_qsv`** are unchanged.
- **The user's `get_format()` callback** is not called for a kept change.
  fftools' returns the same format anyway.

Verified offline only: it applies at `--fuzz=0` to n9.0.1 on its own and after
the four patches above, and `hevcdec.c` compiles without warnings. That the
keep path is taken at the end splices, that the segfault stops, and that the
kept context decodes the same frames as software still needs a run on the
card.
