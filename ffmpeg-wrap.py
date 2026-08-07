#!/usr/bin/env python3
"""Compatibility shim: strip -vsync/-async (removed in ffmpeg master) for av1an."""
import os
import sys

SKIP_FLAGS = {"-vsync", "-async"}


def parse_args(argv):
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in SKIP_FLAGS:
            # next token is the value; skip it
            i += 2
            continue
        if a.startswith("-vsync=") or a.startswith("-async="):
            i += 1
            continue
        if a == "--":
            out.extend(argv[i:])
            break
        out.append(a)
        i += 1
    return out


def main():
    real = "/usr/local/bin/ffmpeg-real"
    args = parse_args(sys.argv[1:])
    os.execv(real, [real] + args)


if __name__ == "__main__":
    import sys

    main()