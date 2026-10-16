"""Frontal dial renderer (PLAN §3 step 1).

Frame: face pixels of a `size` x `size` render, x right, y down, pixel centres at integer coordinates.
Angles are atan2 angles in that frame (clockwise from +x); the scale runs clockwise from min to max.
Drawing is supersampled SS x and box-downsampled, so face px x maps to canvas px SS * x + (SS - 1) / 2.
"""

from __future__ import annotations

import colorsys
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SS = 2  # supersampling factor


@dataclass
class Dial:
    face: np.ndarray            # (S, S, 3) uint8 RGB, valid where alpha > 0
    alpha: np.ndarray           # (S, S) float32 in [0, 1]: dial (face + bezel) coverage
    needle: np.ndarray          # (S, S, 4) uint8 RGBA (straight alpha), needle + pivot cap
    keypoints: np.ndarray       # (7, 2) float64, K0..K6 in face px
    centre: np.ndarray          # (2,) face px; also the pivot
    face_radius: float          # face px, inner edge of the bezel
    dial_radius: float          # face px, outer edge of the bezel
    meta: dict = field(default_factory=dict)


def font_dir() -> str:
    import matplotlib
    return os.path.join(matplotlib.get_data_path(), "fonts", "ttf")


@lru_cache(maxsize=256)
def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.path.join(font_dir(), name), size)


def _choice(rng: np.random.Generator, weights: dict):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _u(rng, lo_hi) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _fmt(v: float) -> str:
    s = f"{round(v, 6):g}"
    return "0" if s == "-0" else s


def _luma(rgb) -> float:
    r, g, b = (c / 255 for c in rgb)
    return 0.299 * r + 0.587 * g + 0.114 * b


class _Canvas:
    """Thin wrapper converting face px to supersampled canvas px."""

    def __init__(self, size: int, mode: str, fill):
        self.img = Image.new(mode, (size * SS, size * SS), fill)
        self.draw = ImageDraw.Draw(self.img)

    @staticmethod
    def p(x, y):
        return (SS * x + (SS - 1) / 2, SS * y + (SS - 1) / 2)

    def disc(self, c, r, fill):
        x, y = self.p(*c)
        self.draw.ellipse([x - SS * r, y - SS * r, x + SS * r, y + SS * r], fill=fill)

    def poly(self, pts, fill):
        self.draw.polygon([self.p(*q) for q in pts], fill=fill)

    def arc(self, c, r_outer, a0_deg, a1_deg, width, fill):
        x, y = self.p(*c)
        R = SS * r_outer
        self.draw.arc([x - R, y - R, x + R, y + R], a0_deg, a1_deg, fill=fill, width=max(1, round(SS * width)))

    def text(self, c, s, font_name, size, fill):
        self.draw.text(self.p(*c), s, font=_font(font_name, max(4, round(SS * size))), fill=fill, anchor="mm")

    def text_width(self, s, font_name, size) -> float:
        l, _, r, _ = _font(font_name, max(4, round(SS * size))).getbbox(s)
        return (r - l) / SS

    def down(self, size: int) -> Image.Image:
        return self.img.resize((size, size), Image.Resampling.BOX)


def _radial_quad(c, a, r0, r1, w0, w1=None):
    """Quad along angle a (rad) from radius r0 to r1, width w0 at r0 and w1 at r1."""
    w1 = w0 if w1 is None else w1
    d = np.array([np.cos(a), np.sin(a)])
    n = np.array([-d[1], d[0]])
    c = np.asarray(c, float)
    return [c + r0 * d + w0 / 2 * n, c + r1 * d + w1 / 2 * n, c + r1 * d - w1 / 2 * n, c + r0 * d - w0 / 2 * n]


def _face_colours(rng, style):
    if style == "light":
        t = rng.uniform(0, 1)
        face = tuple(int(v) for v in np.array([252, 252, 250]) * (1 - t) + np.array([243, 234, 208]) * t)
        ink = tuple([int(rng.integers(10, 45))] * 3)
    elif style == "dark":
        g = int(rng.integers(12, 40))
        face = (g, g, g + int(rng.integers(0, 6)))
        ink = tuple([int(rng.integers(220, 252))] * 3)
    else:
        r, g, b = colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.3, 0.7), rng.uniform(0.45, 0.9))
        face = (int(255 * r), int(255 * g), int(255 * b))
        ink = (15, 15, 15) if _luma(face) > 0.5 else (240, 240, 240)
    return face, ink


def _bezel_shades(rng, style, n):
    """n RGB tuples from outer to inner edge of the bezel."""
    t = np.linspace(0, 1, n)
    if style == "chrome":
        base = rng.uniform(150, 200)
        v = base + 55 * np.cos(2 * np.pi * (t + rng.uniform(0, 1))) * rng.uniform(0.5, 1)
        return [tuple([int(np.clip(x, 60, 250))] * 3) for x in v]
    if style == "brass":
        v = 0.75 + 0.25 * np.cos(2 * np.pi * (t + rng.uniform(0, 1)))
        return [(int(205 * x), int(165 * x), int(70 * x)) for x in v]
    v = 25 + 20 * np.cos(2 * np.pi * (t + rng.uniform(0, 1)))
    return [tuple([int(x)] * 3) for x in v]


def _scale_layout(rng, cfg):
    """Pick range, major intervals and minor divisions such that 25/50/75 % fall on ticks."""
    entry = cfg["ranges"][rng.integers(len(cfg["ranges"]))]
    lo, hi = entry["range"]
    n_major = int(entry["majors"][rng.integers(len(entry["majors"]))])
    m_lo, m_hi = cfg["minor_per_major"]
    options = [m for m in range(m_lo, m_hi + 1) if (n_major * m) % 4 == 0]
    n_minor = int(options[rng.integers(len(options))])
    return float(lo), float(hi), n_major, n_minor


def _sample_f(rng, cfg_value) -> float:
    if rng.uniform() < cfg_value["outside_prob"]:
        m = cfg_value["outside_margin"]
        return float(rng.uniform(-m, 0.0) if rng.uniform() < 0.5 else rng.uniform(1.0 + 1e-9, 1.0 + m))
    return float(rng.uniform(0.0, 1.0))


def _needle_polygons(style, c, a, L, w, rng):
    """List of polygons (face px) for the needle pointing at angle a with its apex exactly at c + L * dir."""
    d = np.array([np.cos(a), np.sin(a)])
    n = np.array([-d[1], d[0]])
    c = np.asarray(c, float)

    def P(local):
        return [c + u * d + v * n for u, v in local]

    # Blunt tip (>= 3 face px wide) so the visible end of the needle is the keypoint, not a sub-pixel sliver.
    tw = max(0.1 * w, 3.0)
    if style == "tapered":
        t = rng.uniform(0.10, 0.25) * L
        return [P([(-t, -0.35 * w), (0, -0.5 * w), (L, -tw / 2), (L, tw / 2), (0, 0.5 * w), (-t, 0.35 * w)])], []
    if style == "arrow":
        t, h, ws = rng.uniform(0.08, 0.2) * L, rng.uniform(0.10, 0.16) * L, 0.35 * w
        shaft = P([(-t, -ws / 2), (L - h, -ws / 2), (L - h, ws / 2), (-t, ws / 2)])
        head = P([(L - h, -0.9 * w), (L, 0.0), (L - h, 0.9 * w)])
        return [shaft, head], []
    t = rng.uniform(0.25, 0.35) * L
    shaft = P([(-t, -0.2 * w), (0, -0.25 * w), (L, -tw / 2), (L, tw / 2), (0, 0.25 * w), (-t, 0.2 * w)])
    weight = (c - 0.8 * t * d, rng.uniform(0.6, 1.0) * w)          # counterweight disc (centre, radius)
    return [shaft], [weight]


def render_dial(rng: np.random.Generator, cfg: dict) -> Dial:
    """Render one frontal dial from the `render` section of the generator config."""
    S = int(cfg["size"])
    c = np.array([(S - 1) / 2, (S - 1) / 2])
    R_dial = cfg["dial_radius_frac"] * S
    bezel_style = _choice(rng, cfg["bezel"]["styles"])
    bw = _u(rng, cfg["bezel"]["width_frac"])
    R = R_dial / (1 + bw)                                   # face radius

    face_style = _choice(rng, cfg["face"]["styles"])
    face_rgb, ink = _face_colours(rng, face_style)

    sc = cfg["scale"]
    sweep = float(sc["sweeps_deg"][rng.choice(len(sc["sweeps_deg"]), p=np.array(sc["sweep_probs"]) / sum(sc["sweep_probs"]))])
    start = -90.0 - sweep / 2 + rng.uniform(-sc["centre_jitter_deg"], sc["centre_jitter_deg"])
    lo, hi, n_major, n_minor = _scale_layout(rng, sc)
    r_t = _u(rng, sc["tick_ring_frac"]) * R
    inward = bool(rng.uniform() < sc["ticks_inward_prob"])
    L_maj = _u(rng, sc["major_len_frac"]) * R
    if not inward:
        L_maj = min(L_maj, 0.96 * R - r_t)
    L_min = min(_u(rng, sc["minor_len_frac"]) * R, 0.7 * L_maj)
    W_maj = _u(rng, sc["major_width_frac"]) * R
    W_min = 0.5 * W_maj

    tx = cfg["text"]
    font = tx["fonts"][rng.integers(len(tx["fonts"]))]
    f = _sample_f(rng, cfg["value"])

    # ---- face layer ---------------------------------------------------------------------------
    cv = _Canvas(S, "RGB", (0, 0, 0))
    shades = _bezel_shades(rng, bezel_style, 24)
    for i, col in enumerate(shades):
        cv.disc(c, R_dial - (R_dial - R) * i / len(shades), col)
    cv.disc(c, R, face_rgb)

    zones = []
    if rng.uniform() < sc["zone_prob"]:
        band = rng.uniform(0.03, 0.06) * R
        r_band = r_t if inward else r_t - 0.005 * R
        red0 = rng.uniform(0.7, 0.9)
        zones.append(("red", red0, 1.0))
        if rng.uniform() < 0.5:
            g0 = rng.uniform(0.2, 0.4)
            zones.append(("green", g0, g0 + rng.uniform(0.2, 0.3)))
        for name, z0, z1 in zones:
            col = (200, 30, 30) if name == "red" else (40, 160, 60)
            cv.arc(c, r_band, start + z0 * sweep, start + z1 * sweep, band, col)

    if rng.uniform() < sc["arc_line_prob"]:
        lw = max(0.6 * W_min, 1.0)
        cv.arc(c, r_t + lw / 2, start, start + sweep, lw, ink)

    n_ticks = n_major * n_minor
    for i in range(n_ticks + 1):
        a = np.radians(start + sweep * i / n_ticks)
        major = i % n_minor == 0
        Lt, Wt = (L_maj, W_maj) if major else (L_min, W_min)
        r0, r1 = (r_t - Lt, r_t) if inward else (r_t, r_t + Lt)
        cv.poly(_radial_quad(c, a, r0, r1, Wt), ink)

    # numerals at major ticks, shrunk if neighbours would overlap
    labels = [_fmt(lo + (hi - lo) * k / n_major) for k in range(n_major + 1)]
    size = _u(rng, tx["numeral_size_frac"]) * R
    gap = 0.05 * R
    r_num = r_t - (L_maj if inward else 0.0) - gap - 0.6 * size
    spacing = r_num * np.radians(sweep) / n_major
    widest = max(cv.text_width(s, font, size) for s in labels)
    size *= min(1.0, 0.8 * spacing / max(widest, 1e-6))
    r_num = r_t - (L_maj if inward else 0.0) - gap - 0.6 * size
    for k, s in enumerate(labels):
        a = np.radians(start + sweep * k / n_major)
        cv.text(c + r_num * np.array([np.cos(a), np.sin(a)]), s, font, size, ink)

    unit = tx["units"][rng.integers(len(tx["units"]))]
    cv.text(c + np.array([0.0, 0.32 * R]), unit, font, rng.uniform(0.08, 0.11) * R, ink)
    brand = None
    if rng.uniform() < tx["brand_prob"]:
        brand = tx["brands"][rng.integers(len(tx["brands"]))]
        cv.text(c + np.array([0.0, -0.30 * R]), brand, font, rng.uniform(0.05, 0.08) * R, ink)

    face = np.asarray(cv.down(S), np.uint8)

    am = _Canvas(S, "L", 0)
    am.disc(c, R_dial, 255)
    alpha = np.asarray(am.down(S), np.float32) / 255.0

    # ---- needle layer -------------------------------------------------------------------------
    nd = cfg["needle"]
    n_style = _choice(rng, nd["styles"])
    light_face = _luma(face_rgb) > 0.5
    colours = {k: v for k, v in nd["colours"].items() if not (k == "white" and light_face)}
    n_colour = _choice(rng, colours)
    rgb = {"black": (20, 20, 22), "red": (205, 25, 25), "white": (238, 238, 235)}[n_colour]
    L = _u(rng, nd["length_frac"]) * r_t
    w = _u(rng, nd["width_frac"]) * R
    a_tip = np.radians(start + f * sweep)
    polys, discs = _needle_polygons(n_style, c, a_tip, L, w, rng)

    nl = _Canvas(S, "RGBA", (0, 0, 0, 0))
    for pts in polys:
        nl.poly(pts, rgb + (255,))
    for dc, dr in discs:
        nl.disc(dc, dr, rgb + (255,))
    cap_r = _u(rng, nd["cap_frac"]) * R
    cap_rgb = rgb if rng.uniform() < 0.5 else tuple([int(rng.integers(30, 190))] * 3)
    nl.disc(c, cap_r, cap_rgb + (255,))
    nl.disc(c - 0.25 * cap_r, 0.3 * cap_r, tuple(min(255, v + 50) for v in cap_rgb) + (255,))
    needle = np.asarray(nl.img.convert("RGBa").resize((S, S), Image.Resampling.BOX).convert("RGBA"), np.uint8)

    # ---- keypoints (K0..K6, face px) ------------------------------------------------------------
    def on_ring(q):
        a = np.radians(start + q * sweep)
        return c + r_t * np.array([np.cos(a), np.sin(a)])

    kp = np.array([c, c + L * np.array([np.cos(a_tip), np.sin(a_tip)]), *[on_ring(q) for q in (0, 0.25, 0.5, 0.75, 1)]])

    meta = {
        "f": f, "value": lo + f * (hi - lo), "range": [lo, hi], "units": unit,
        "sweep_deg": sweep, "start_deg": float(start),
        "style": {
            "face": face_style, "bezel": bezel_style, "needle": n_style, "needle_colour": n_colour,
            "font": font, "n_major": n_major, "n_minor": n_minor, "ticks_inward": inward,
            "zones": [z[0] for z in zones], "brand": brand,
        },
    }
    return Dial(face=face, alpha=alpha, needle=needle, keypoints=kp, centre=c,
                face_radius=float(R), dial_radius=float(R_dial), meta=meta)
