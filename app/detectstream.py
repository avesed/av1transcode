"""A PySceneDetect frame source fed straight from an ffmpeg pipe.

Scene detection used to run as two passes over the whole file: ffmpeg wrote a
downscaled copy, then OpenCV decoded that copy again frame by frame. Measured
on an 8-minute 4K window, that is 85.7s of ffmpeg followed by 51.1s of OpenCV.

Decoding the 4K source is 81.3s of the first number - the scale filter and the
x264 encode together add 4.4s - so the copy is almost entirely decode, and the
second pass repeats work that has already been done. Piping the frames instead
lets detection run WHILE ffmpeg decodes, which takes the phase down to roughly
the decode alone, and writes no intermediate file (a 45-minute 4K episode
staged around 560MB).

Imported lazily: PySceneDetect is an optional dependency of the optimizer.
"""

from __future__ import annotations

import re
import subprocess
import threading
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_stream import VideoStream

# The `W:H` form of ffmpeg's scale filter, with -1/-2 meaning "derive from the
# other side". Anything more elaborate (an expression, a named option) is left
# to the caller to handle, because the frame size has to be known up front to
# read fixed-size frames off a rawvideo pipe.
_SCALE_WH = re.compile(r"^\s*(-?\d+)\s*:\s*(-?\d+)\s*$")


def scaled_size(spec: str, src_w: int, src_h: int) -> Optional[Tuple[int, int]]:
    """Explicit (width, height) a `W:H` scale spec produces, or None.

    None means "cannot know without asking ffmpeg", which is the signal to fall
    back to the file-based path rather than guess: a wrong size would silently
    shear every frame, and a sheared frame still detects *something*.
    """
    m = _SCALE_WH.match(spec or "")
    if not m or src_w <= 0 or src_h <= 0:
        return None
    sw, sh = int(m.group(1)), int(m.group(2))
    if sw > 0 and sh > 0:
        return sw, sh
    if sw > 0:                                  # height derived from width
        mult = abs(sh) if sh < 0 else 1
        h = max(mult, round(src_h * sw / src_w / mult) * mult)
        return sw, h
    if sh > 0:                                  # width derived from height
        mult = abs(sw) if sw < 0 else 1
        w = max(mult, round(src_w * sh / src_h / mult) * mult)
        return w, sh
    return None


class PipedFrames(VideoStream):
    """Fixed-size BGR frames read off an ffmpeg rawvideo pipe.

    Only what SceneManager touches is implemented: it reads forward to the end
    and never seeks, so seek/reset raise rather than pretending.
    """

    def __init__(self, args: List[str], width: int, height: int,
                 fps: float, total_frames: int, path: str) -> None:
        self._w, self._h = int(width), int(height)
        self._frame_bytes = self._w * self._h * 3     # bgr24
        self._fps = Fraction(fps).limit_denominator(1000000)
        self._total = max(1, int(total_frames))
        self._path = path
        self._n = 0
        self._closed = False
        self.proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        # Drain stderr on a thread, keeping only the tail. Left unread it would
        # fill its pipe and deadlock ffmpeg mid-file, which on this path does
        # not look like a failure - it looks like the video ending early.
        self._err = b""
        self._err_thread = threading.Thread(target=self._drain, daemon=True)
        self._err_thread.start()

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for chunk in iter(lambda: self.proc.stderr.read(4096), b""):
            self._err = (self._err + chunk)[-4096:]

    def check_ok(self) -> Optional[str]:
        """ffmpeg's failure message, or None if it finished cleanly.

        A short read is the ONLY way this stream signals end-of-video, so a
        decoder that dies halfway through is indistinguishable from a video
        that simply ended - and the caller would take the truncated shot list
        as authoritative and encode a fraction of the film. Hence an explicit
        check rather than trusting EOF.
        """
        if self._closed:
            return None
        try:
            rc = self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            return "the decoder did not exit after the last frame"
        if rc == 0:
            return None
        self._err_thread.join(timeout=5)
        tail = self._err.decode("utf-8", "replace").strip()
        return f"the decoder exited {rc}" + (f": {tail[-500:]}" if tail else "")

    # ---- the frame source ----
    def read(self, decode: bool = True):
        assert self.proc.stdout is not None
        buf = self.proc.stdout.read(self._frame_bytes)
        if buf is None or len(buf) < self._frame_bytes:
            return False
        self._n += 1
        if not decode:
            return True
        # A read-only view is enough - the detector converts colour spaces and
        # never writes back - and it saves a 1.5MB memcpy per frame.
        return np.frombuffer(buf, dtype=np.uint8).reshape(self._h, self._w, 3)

    def close(self) -> None:
        self._closed = True
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    # ---- what SceneManager reads ----
    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._w, self._h

    @property
    def frame_rate(self) -> Fraction:
        return self._fps

    @property
    def frame_number(self) -> int:
        return self._n

    @property
    def position(self) -> FrameTimecode:
        return FrameTimecode(max(0, self._n - 1), self._fps)

    @property
    def position_ms(self) -> float:
        return max(0, self._n - 1) * 1000.0 / float(self._fps)

    @property
    def duration(self) -> FrameTimecode:
        return FrameTimecode(self._total, self._fps)

    @property
    def aspect_ratio(self) -> float:
        return 1.0

    @property
    def is_seekable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "ffmpeg-pipe"

    @property
    def path(self) -> str:
        return self._path

    def seek(self, target) -> None:
        raise NotImplementedError("the detection pipe only reads forward")

    def reset(self) -> None:
        raise NotImplementedError("the detection pipe only reads forward")
