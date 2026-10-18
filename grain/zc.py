"""Zero-copy source reads for grain_apply: zcdec decodes on VA-API and passes each kept frame's surface as DMA-BUF
fds; they are imported into torch's own Level Zero context (zcshim) and de-tiled on the card into the same int16
10-bit Y, U, V planes the CPU path's reader_raw() + planes() give. No pixels cross PCIe or system memory.

ZcReader(path, w, h, step, ranges) yields (n, (Y, U, V)) for the kept frames, n in decode output order = the frame
numbers of ffmpeg's -fps_mode passthrough reads. Only 10-bit 4:2:0 (P010 surfaces) for now; anything else raises
ZcUnsupported before the first frame, and the caller reads on the CPU instead."""
import collections, ctypes, os, socket, struct, subprocess
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ZCDEC = os.environ.get("GRAIN_ZCDEC", "/usr/local/bin/zcdec")
SHIM = os.environ.get("GRAIN_ZCSHIM", f"{HERE}/zcdec/libzcshim.so")
INFLIGHT = int(os.environ.get("GRAIN_ZC_INFLIGHT", "4"))

MOD_LINEAR, MOD_4_TILED = 0, 0x0100000000000009       # DRM format modifiers (I915_FORMAT_MOD_4_TILED: Tile4)
ZC_P010, ZC_NV12 = 1, 2                                   # zcdec's surface format codes

# struct zc_msg in zcdec.c (natural alignment, 616 bytes)
HDR = struct.Struct("<6i4Q4Qi4x")
LAYER = struct.Struct("<Ii4i4q4q")
TAIL = struct.Struct("<q160s")
MSG_SIZE = HDR.size + 4 * LAYER.size + TAIL.size


class ZcUnsupported(RuntimeError):
    pass


class ZcError(RuntimeError):
    pass


def _parse(buf):
    if len(buf) != MSG_SIZE:
        raise ZcError(f"zcdec message of {len(buf)} bytes, expected {MSG_SIZE}")
    n, w, h, fmt, bits, nobj, *rest = HDR.unpack_from(buf, 0)
    size, mod, nlay = rest[0:4], rest[4:8], rest[8]
    layers = []
    for i in range(4):
        f, npl, *v = LAYER.unpack_from(buf, HDR.size + i * LAYER.size)
        layers.append({"format": f, "nb_planes": npl, "object": v[0:4], "offset": v[4:8], "pitch": v[8:12]})
    pts, err = TAIL.unpack_from(buf, HDR.size + 4 * LAYER.size)
    return {"n": n, "w": w, "h": h, "fmt": fmt, "bits": bits, "nobj": nobj, "size": size[:nobj], "mod": mod[:nobj],
            "layers": layers[:nlay], "pts": pts, "err": err.split(b"\0")[0].decode(errors="replace")}


class _Shim:
    lib, ctx, dev = None, None, None

    @classmethod
    def get(cls):
        if cls.lib is None:
            lib = ctypes.CDLL(SHIM)
            lib.zc_native.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
            lib.zc_import.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_void_p)]
            lib.zc_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.zc_dlpack.restype = ctypes.c_void_p
            lib.zc_dlpack.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32]
            ctx, dev = ctypes.c_void_p(), ctypes.c_void_p()
            if lib.zc_native(ctypes.c_void_p(torch.xpu.current_stream().sycl_queue), ctypes.byref(ctx), ctypes.byref(dev)):
                raise ZcUnsupported("no Level Zero handles behind torch's queue")
            cls.lib, cls.ctx, cls.dev = lib, ctx, dev
        return cls


_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def _tensor(ptr, nbytes):
    """a torch uint8 tensor over imported device memory (not owned: freed by zc_free)."""
    s = _Shim.get()
    m = s.lib.zc_dlpack(ptr, nbytes, torch.xpu.current_device())
    return torch.from_dlpack(_PyCapsule_New(m, b"dltensor", None))


_IDX = {}


def _tile4_index(rows, row_bytes, pitch, elem, dev):
    """element index (units of elem bytes, from the plane's offset) of each (row, column) sample of a Tile4 plane:
    4 KB tiles of 128 bytes x 32 rows, laid out row-major at pitch / 128 tiles per row; inside a tile the byte
    address interleaves x[3:0] | y[1:0] | x[5:4] | y[2] | x[6] | y[3] | y[4] (vmafx dmabuf_import.cpp)."""
    key = (rows, row_bytes, pitch, elem, dev)
    if key not in _IDX:
        y = torch.arange(rows, device=dev, dtype=torch.int64)[:, None]
        bx = (torch.arange(row_bytes // elem, device=dev, dtype=torch.int64) * elem)[None, :]
        ix, iy = bx & 127, y & 31
        off = ((ix & 15) | ((iy & 3) << 4) | (((ix >> 4) & 3) << 6) | (((iy >> 2) & 1) << 8) | (((ix >> 6) & 1) << 9)
               | (((iy >> 3) & 1) << 10) | (((iy >> 4) & 1) << 11))
        _IDX[key] = (((y // 32) * (pitch // 128) + bx // 128) * 4096 + off) // elem
    return _IDX[key]


def _plane16(buf, offset, pitch, rows, row_bytes, mod):
    """a 16-bit plane out of an imported buffer (uint8 tensor) as int16 (rows, row_bytes / 2)."""
    if offset % 2:
        raise ZcUnsupported("odd plane offset")
    b16 = buf.view(torch.int16)
    if mod == MOD_4_TILED:
        if pitch % 128 or offset % 4096:
            raise ZcUnsupported(f"Tile4 plane with pitch {pitch} / offset {offset}")
        return b16[offset // 2:][_tile4_index(rows, row_bytes, pitch, 2, buf.device)]
    if mod == MOD_LINEAR:
        return b16[offset // 2:offset // 2 + rows * pitch // 2].view(rows, pitch // 2)[:, :row_bytes // 2]
    raise ZcUnsupported(f"surface modifier {mod:#x}")


class ZcReader:
    """the kept frames of path's first video stream, decoded on VA-API, as int16 (Y, U, V) on torch's device."""

    def __init__(self, path, w, h, step, ranges=(), node=None):
        if not (os.path.exists(ZCDEC) and os.path.exists(SHIM)):
            raise ZcUnsupported("zcdec / zcshim not installed")
        _Shim.get()
        self.w, self.h = w, h
        self.imports = {}                        # dma-buf inode -> (ptr, uint8 tensor)
        self.sock, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        env = dict(os.environ, ZC_INFLIGHT=str(INFLIGHT))
        if node:
            env["RENDER_NODE"] = node
        args = [ZCDEC, path, str(child.fileno()), str(step)] + [f"{a}:{b}" for a, b in ranges]
        self.proc = subprocess.Popen(args, pass_fds=[child.fileno()], env=env, stdin=subprocess.DEVNULL)
        child.close()
        self.decoded = None

    def _import(self, fd, size):
        ino = os.fstat(fd).st_ino
        if ino not in self.imports:
            s = _Shim.get()
            ptr = ctypes.c_void_p()
            r = s.lib.zc_import(s.ctx, s.dev, fd, size, ctypes.byref(ptr))
            if r or not ptr.value:
                raise ZcUnsupported(f"Level Zero DMA-BUF import failed ({r:#x})")
            self.imports[ino] = (ptr, _tensor(ptr, size))
        return self.imports[ino][1]

    def _planes(self, m, bufs):
        if m["fmt"] != ZC_P010 or m["bits"] != 10:
            raise ZcUnsupported(f"surface format {m['fmt']} / {m['bits']} bits (P010 only)")
        if (m["w"], m["h"]) != (self.w, self.h):
            raise ZcError(f"surface {m['w']}x{m['h']}, expected {self.w}x{self.h}")
        if len(m["layers"]) == 2:
            (ly, luv) = m["layers"]
            py = (ly["object"][0], ly["offset"][0], ly["pitch"][0])
            puv = (luv["object"][0], luv["offset"][0], luv["pitch"][0])
        elif len(m["layers"]) == 1 and m["layers"][0]["nb_planes"] == 2:
            l0 = m["layers"][0]
            py, puv = ((l0["object"][i], l0["offset"][i], l0["pitch"][i]) for i in (0, 1))
        else:
            raise ZcUnsupported(f"surface layout {m['layers']}")
        w, h = self.w, self.h
        y = _plane16(bufs[py[0]], py[1], py[2], h, 2 * w, m["mod"][py[0]])
        uv = _plane16(bufs[puv[0]], puv[1], puv[2], h // 2, 2 * w, m["mod"][puv[0]])
        # P010 keeps the 10 bits at the top of each 16; >> 6 on int16 sign-extends, & 1023 takes the value back
        y = (y >> 6) & 1023
        uv = (uv >> 6) & 1023
        return y.contiguous(), uv[:, 0::2].contiguous(), uv[:, 1::2].contiguous()

    def __iter__(self):
        pending = collections.deque()
        try:
            while True:
                msg, fds, _flags, _addr = socket.recv_fds(self.sock, MSG_SIZE, 8)
                try:
                    if not msg:
                        raise ZcError(f"zcdec ended without a word (rc {self.proc.poll()})")
                    m = _parse(msg)
                    if m["n"] == -2:
                        raise ZcUnsupported(f"zcdec: {m['err']}")
                    if m["n"] == -1:
                        self.decoded = m["pts"]
                        break
                    bufs = [self._import(fd, sz) for fd, sz in zip(fds, m["size"])]
                finally:
                    for fd in fds:
                        try:
                            os.close(fd)
                        except OSError:             # the Level Zero import may already have taken (closed) it
                            pass
                planes = self._planes(m, bufs)
                ev = torch.xpu.Event()
                ev.record()
                pending.append(ev)
                while len(pending) >= INFLIGHT:     # the surface goes back to the decoder once its copy is done
                    pending.popleft().synchronize()
                    self.sock.send(b"a")
                yield m["n"], planes
            while pending:
                pending.popleft().synchronize()
                self.sock.send(b"a")
        finally:
            self.close()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.sock.close()
        if self.imports:
            torch.xpu.synchronize()
            s = _Shim.get()
            for ptr, _t in self.imports.values():
                s.lib.zc_free(s.ctx, ptr)
            self.imports.clear()
