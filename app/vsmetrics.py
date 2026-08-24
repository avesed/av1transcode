"""SSIMULACRA2 scoring through VapourSynth, run as a subprocess.

libvmaf cannot compute SSIMULACRA2 (2.3.1 has no such model, and the
ssimulacra2_rs CLI is pinned to VapourSynth's removed API-3), so the metric is
reached the way the encoding community reaches it: the vszip plugin, fed by
bestsource. Both are loaded here rather than in the optimizer because a
VapourSynth core is process-global and the optimizer runs a thread pool.

Usage:
    python3 -m app.vsmetrics REF DIST --step N [--matrix M --transfer T
                                                --primaries P --range R]
prints the mean score to stdout.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vsmetrics")
    ap.add_argument("reference")
    ap.add_argument("distorted")
    ap.add_argument("--step", type=int, default=1,
                    help="score every Nth frame (still-image metric, so a "
                         "subset is sound; the ENCODE is never subsampled)")
    ap.add_argument("--matrix", default="709")
    ap.add_argument("--transfer", default="709")
    ap.add_argument("--primaries", default="709")
    ap.add_argument("--range", dest="crange", default="limited")
    ap.add_argument("--vszip", default="/usr/local/lib/vapoursynth/libvszip.so")
    ap.add_argument("--bestsource",
                    default="/usr/local/lib/vapoursynth/libbestsource.so")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)

    import vapoursynth as vs

    core = vs.core
    if args.threads > 0:
        core.num_threads = args.threads
    def load(path: str, namespace: str) -> None:
        # VapourSynth auto-loads everything in its plugin dir, so an explicit
        # load of an already-present plugin raises. Only load what is missing.
        if hasattr(core, namespace):
            return
        core.std.LoadPlugin(path)

    load(args.vszip, "vszip")
    load(args.bestsource, "bs")

    def source(path: str):
        # bestsource indexes a whole file before it can serve frames, so both
        # inputs must be small - the optimizer hands it a per-shot reference
        # shard and the probe's own ivf, never the multi-GB source.
        return core.bs.VideoSource(path)

    def to_linear_rgb(clip):
        # SSIMULACRA2 is defined on linear-light RGB; the source is coded
        # (PQ/HLG/gamma) so the transfer has to be undone, not just relabelled.
        return core.resize.Bicubic(
            clip, format=vs.RGBS,
            matrix_in_s=args.matrix, transfer_in_s=args.transfer,
            primaries_in_s=args.primaries, range_in_s=args.crange,
            transfer_s="linear", primaries_s=args.primaries)

    ref, dist = source(args.reference), source(args.distorted)
    n = min(len(ref), len(dist))
    if n == 0:
        print("no frames to score", file=sys.stderr)
        return 2
    ref, dist = ref[:n], dist[:n]
    step = max(1, args.step)
    if step > 1:
        ref = core.std.SelectEvery(ref, cycle=step, offsets=0)
        dist = core.std.SelectEvery(dist, cycle=step, offsets=0)

    scored = core.vszip.SSIMULACRA2(to_linear_rgb(ref), to_linear_rgb(dist))
    values = [f.props["SSIMULACRA2"] for f in scored.frames()]
    if not values:
        print("scored no frames", file=sys.stderr)
        return 2
    print(f"{sum(values) / len(values):.6f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
