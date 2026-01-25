import os, struct, zlib, math

# ---------- minimal bitmap font (5x7) ----------
# Each glyph is 7 rows of 5 bits (as strings).
FONT_5x7 = {
    "A": ["01110","10001","10001","11111","10001","10001","10001"],
    "B": ["11110","10001","10001","11110","10001","10001","11110"],
    "C": ["01111","10000","10000","10000","10000","10000","01111"],
    "D": ["11110","10001","10001","10001","10001","10001","11110"],
    "E": ["11111","10000","10000","11110","10000","10000","11111"],
    "F": ["11111","10000","10000","11110","10000","10000","10000"],
    "G": ["01111","10000","10000","10111","10001","10001","01111"],
    "H": ["10001","10001","10001","11111","10001","10001","10001"],
    "I": ["11111","00100","00100","00100","00100","00100","11111"],
    "K": ["10001","10010","10100","11000","10100","10010","10001"],
    "L": ["10000","10000","10000","10000","10000","10000","11111"],
    "M": ["10001","11011","10101","10101","10001","10001","10001"],
    "N": ["10001","11001","10101","10011","10001","10001","10001"],
    "O": ["01110","10001","10001","10001","10001","10001","01110"],
    "P": ["11110","10001","10001","11110","10000","10000","10000"],
    "Q": ["01110","10001","10001","10001","10101","10010","01101"],
    "R": ["11110","10001","10001","11110","10100","10010","10001"],
    "S": ["01111","10000","10000","01110","00001","00001","11110"],
    "T": ["11111","00100","00100","00100","00100","00100","00100"],
    "U": ["10001","10001","10001","10001","10001","10001","01110"],
    "V": ["10001","10001","10001","10001","01010","01010","00100"],
    "W": ["10001","10001","10001","10101","10101","11011","10001"],
    "X": ["10001","01010","00100","00100","00100","01010","10001"],
    "Y": ["10001","01010","00100","00100","00100","00100","00100"],
    "Z": ["11111","00001","00010","00100","01000","10000","11111"],
    "0": ["01110","10001","10011","10101","11001","10001","01110"],
    "1": ["00100","01100","00100","00100","00100","00100","01110"],
    "2": ["01110","10001","00001","00010","00100","01000","11111"],
    "3": ["11110","00001","00001","01110","00001","00001","11110"],
    "4": ["00010","00110","01010","10010","11111","00010","00010"],
    "5": ["11111","10000","10000","11110","00001","00001","11110"],
    "6": ["01110","10000","10000","11110","10001","10001","01110"],
    "7": ["11111","00001","00010","00100","01000","01000","01000"],
    "8": ["01110","10001","10001","01110","10001","10001","01110"],
    "9": ["01110","10001","10001","01111","00001","00001","01110"],
    ".": ["00000","00000","00000","00000","00000","00100","00100"],
    ":": ["00000","00100","00100","00000","00100","00100","00000"],
    "-": ["00000","00000","00000","11111","00000","00000","00000"],
    " ": ["00000","00000","00000","00000","00000","00000","00000"],
    "/": ["00001","00010","00100","01000","10000","00000","00000"],
    "(" : ["00010","00100","01000","01000","01000","00100","00010"],
    ")" : ["01000","00100","00010","00010","00010","00100","01000"],
}

def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

def _png_write_rgb(path: str, w: int, h: int, rgb: bytes) -> None:
    assert len(rgb) == w * h * 3
    raw = bytearray()
    stride = w * 3
    for y in range(h):
        raw.append(0)
        raw += rgb[y*stride:(y+1)*stride]
    comp = zlib.compress(bytes(raw), 9)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", comp))
        f.write(_chunk(b"IEND", b""))

def _draw_text(rgb, W, H, x, y, text, color=(0,0,0), scale=2):
    # top-left origin
    def put(px, py):
        if 0 <= px < W and 0 <= py < H:
            i = (py*W + px)*3
            rgb[i:i+3] = bytes(color)
    cx = x
    for ch in text:
        ch = ch.upper()
        glyph = FONT_5x7.get(ch, FONT_5x7[" "])
        for row in range(7):
            bits = glyph[row]
            for col in range(5):
                if bits[col] == "1":
                    for sy in range(scale):
                        for sx in range(scale):
                            put(cx + col*scale + sx, y + row*scale + sy)
        cx += (6*scale)  # 5 + spacing

def _line(rgb, W, H, x0, y0, x1, y1, color=(0,0,0)):
    # Bresenham
    dx = abs(x1-x0)
    dy = -abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < W and 0 <= y0 < H:
            i = (y0*W + x0)*3
            rgb[i:i+3] = bytes(color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2*err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

def _dot(rgb, W, H, x, y, color, r=5):
    for dx in range(-r, r+1):
        for dy in range(-r, r+1):
            if dx*dx + dy*dy <= r*r:
                px, py = x+dx, y+dy
                if 0 <= px < W and 0 <= py < H:
                    i = (py*W + px)*3
                    rgb[i:i+3] = bytes(color)

def write_qg_screening_plot(alphaA: dict, alphaB: dict, out_path: str,
                            eps0: float = None, g_eff: float = None, max_rel_delta: float = None) -> None:
    labels = ["Mercury", "DoublePulsar", "BH_proxy"]
    x_labels = ["Mercury", "Double Pulsar", "BH proxy"]

    valsA = [float(alphaA[k]) for k in labels]
    valsB = [float(alphaB.get(k, float("nan"))) for k in labels]

    # Canvas
    W, H = 1200, 700
    bg = (255,255,255)
    rgb = bytearray(bg * (W*H))

    # Layout
    margin_l, margin_r = 110, 40
    margin_t, margin_b = 90, 120
    x0, x1 = margin_l, W - margin_r
    y0, y1 = margin_t, H - margin_b

    # Title
    _draw_text(rgb, W, H, 60, 20, "DEMO-66: Quantum Gravity Screening Witnesses", (0,0,0), scale=2)

    # Compute log scale mapping
    allv = [v for v in valsA if v > 0] + [v for v in valsB if v > 0 and v == v]
    vmin = min(allv)
    vmax = max(allv)

    lmin = math.floor(math.log10(vmin))
    lmax = math.ceil(math.log10(vmax))

    def ymap(v):
        lv = math.log10(max(v, 10**lmin))
        t = (lv - lmin) / (lmax - lmin) if lmax != lmin else 0.5
        return int(y1 - t*(y1-y0))

    xs = [x0 + int((i+0.5)*(x1-x0)/3) for i in range(3)]

    # Axes
    _line(rgb, W, H, x0, y1, x1, y1, (0,0,0))
    _line(rgb, W, H, x0, y0, x0, y1, (0,0,0))

    # Grid + y ticks with labels
    for p in range(lmin, lmax+1):
        y = ymap(10**p)
        # gridline
        _line(rgb, W, H, x0, y, x1, y, (235,235,235))
        # tick
        _line(rgb, W, H, x0-6, y, x0, y, (0,0,0))
        _draw_text(rgb, W, H, 18, y-7, f"1e{p}", (0,0,0), scale=2)

    # Axis labels
    _draw_text(rgb, W, H, 420, H-60, "Test body", (0,0,0), scale=2)
    _draw_text(rgb, W, H, 20, 60, "alpha (screening strength)  log10 scale", (0,0,0), scale=2)

    # X tick marks + labels
    for x, lab in zip(xs, x_labels):
        _line(rgb, W, H, x, y1, x, y1+10, (0,0,0))
        _draw_text(rgb, W, H, x-70, y1+20, lab, (0,0,0), scale=2)

    # Plot points
    blue = (0,90,200)
    red  = (200,30,30)

    for x, v in zip(xs, valsA):
        _dot(rgb, W, H, x, ymap(v), blue, r=6)
    for x, v in zip(xs, valsB):
        if v != v:  # nan
            continue
        _dot(rgb, W, H, x, ymap(v), red, r=6)

    # Legend box with text
    lx, ly = W-420, y0 + 20
    _line(rgb, W, H, lx, ly, lx+360, ly, (0,0,0))
    _line(rgb, W, H, lx, ly+120, lx+360, ly+120, (0,0,0))
    _line(rgb, W, H, lx, ly, lx, ly+120, (0,0,0))
    _line(rgb, W, H, lx+360, ly, lx+360, ly+120, (0,0,0))

    _dot(rgb, W, H, lx+25, ly+35, blue, r=6)
    _draw_text(rgb, W, H, lx+50, ly+25, "Witness A (piecewise)", (0,0,0), scale=2)

    _dot(rgb, W, H, lx+25, ly+80, red, r=6)
    _draw_text(rgb, W, H, lx+50, ly+70, "Witness B (smooth)", (0,0,0), scale=2)

    # Annotation box (numbers)
    ax, ay = 60, y0 + 10
    _draw_text(rgb, W, H, ax, ay, "Key numbers:", (0,0,0), scale=2)
    if eps0 is not None:
        _draw_text(rgb, W, H, ax, ay+30, f"eps0 = {eps0:.3e}", (0,0,0), scale=2)
    if g_eff is not None:
        _draw_text(rgb, W, H, ax, ay+60, f"g_eff = {g_eff:.3e}", (0,0,0), scale=2)
    if max_rel_delta is not None:
        _draw_text(rgb, W, H, ax, ay+90, f"max rel delta(A,B) = {max_rel_delta:.3e}", (0,0,0), scale=2)

    _png_write_rgb(out_path, W, H, bytes(rgb))
