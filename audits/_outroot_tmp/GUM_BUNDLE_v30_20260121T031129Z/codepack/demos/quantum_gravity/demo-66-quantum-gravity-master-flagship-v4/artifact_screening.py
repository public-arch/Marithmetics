import os, struct, zlib, math

def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

def _png_write_rgb(path: str, w: int, h: int, rgb: bytes) -> None:
    assert len(rgb) == w * h * 3
    raw = bytearray()
    stride = w * 3
    for y in range(h):
        raw.append(0)  # filter 0
        raw += rgb[y*stride:(y+1)*stride]
    comp = zlib.compress(bytes(raw), 9)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", comp))
        f.write(_chunk(b"IEND", b""))

def write_qg_screening_plot(alphaA: dict, alphaB: dict, out_path: str) -> None:
    # Deterministic points for A/B screening witnesses at three bodies.
    labels = ["Mercury", "DoublePulsar", "BH_proxy"]
    valsA = [float(alphaA[k]) for k in labels]
    valsB = [float(alphaB.get(k, "nan")) for k in labels]

    w, h = 900, 320
    bg = (255, 255, 255)
    rgb = bytearray(bg * (w * h))

    x0, x1 = 80, w - 30
    y0, y1 = 40, h - 60

    def put(x, y, c):
        if 0 <= x < w and 0 <= y < h:
            i = (y * w + x) * 3
            rgb[i:i+3] = bytes(c)

    # axes
    for x in range(x0, x1 + 1):
        put(x, y1, (0, 0, 0))
    for y in range(y0, y1 + 1):
        put(x0, y, (0, 0, 0))

    allv = [v for v in valsA if v > 0] + [v for v in valsB if v > 0 and v == v]
    vmin = min(allv)
    vmax = max(allv)

    def ymap(v):
        lv = math.log10(max(v, vmin))
        l0 = math.log10(vmin)
        l1 = math.log10(vmax)
        t = 0.0 if l1 == l0 else (lv - l0) / (l1 - l0)
        return int(y1 - t * (y1 - y0))

    xs = [x0 + int((i + 0.5) * (x1 - x0) / 3) for i in range(3)]

    # A = blue points
    for x, v in zip(xs, valsA):
        y = ymap(v)
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx*dx + dy*dy <= 10:
                    put(x + dx, y + dy, (0, 90, 200))

    # B = red points (skip NaN)
    for x, v in zip(xs, valsB):
        if v != v:
            continue
        y = ymap(v)
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx*dx + dy*dy <= 10:
                    put(x + dx, y + dy, (200, 30, 30))

    _png_write_rgb(out_path, w, h, bytes(rgb))
