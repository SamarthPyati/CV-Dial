"""Read the printed scale range (min, max) from a dial's numerals with RapidOCR. DEMO ONLY.

Never used by the experiments: CLAUDE.md requires "no OCR anywhere in the experiment path", and eval.evaluate
does not import this module. Install the optional extra to enable it: `uv sync --extra ocr`.

Method (frames: photo px -> upright dial px via pipeline.infer.upright_matrix):
1. Warp the full-resolution photo to an upright, rectified dial (50 % mark up), so numerals are horizontal.
2. OCR enlarged patches around each scale keypoint (K2..K6), inside and outside the ring; keep numbers only.
3. Place each numeral on the scale by its angle around the pivot: q = needle_fraction(pivot, numeral, K2, K6).
4. Fit value = lo + q (hi - lo) robustly (pairs + inliers). A result needs >= min_numerals agreeing numerals.
   It is "read" when the numerals at both ends were themselves read, "inferred" when an end was extrapolated.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from geometry.reading import apply_h, needle_fraction
from pipeline.infer import upright_matrix

NUMBER = re.compile(r"-?\d+(\.\d+)?")


@dataclass
class ScaleRange:
    lo: float
    hi: float
    read_ends: bool                                   # both end numerals read (vs. extrapolated)
    numerals: list[tuple[float, float, float]] = field(default_factory=list)   # (q, value, conf) that agree

    @property
    def how(self) -> str:
        return "read" if self.read_ends else "inferred"


def available() -> bool:
    import importlib.util
    return importlib.util.find_spec("rapidocr") is not None


def parse_number(text: str) -> float | None:
    t = text.strip().replace("−", "-").replace(",", ".").strip(".")
    return float(t) if NUMBER.fullmatch(t) and len(t) <= 6 else None


def fit_range(points: list[tuple[float, float, float]], min_numerals: int = 3, tol: float = 0.02,
              ends_window: float = 0.06, min_spread: float = 0.45) -> ScaleRange | None:
    """Robust fit of value = lo + q (hi - lo) to (q, value, conf) numerals; None if not enough agree."""
    best = None
    for (q1, v1, _), (q2, v2, _) in itertools.combinations(points, 2):
        if abs(q2 - q1) < 0.2 or v1 == v2:
            continue
        b = (v2 - v1) / (q2 - q1)
        a = v1 - b * q1
        inliers = [p for p in points if abs(a + b * p[0] - p[1]) <= tol * abs(b)]
        qs = [p[0] for p in inliers]
        key = (len(inliers), max(qs) - min(qs), sum(p[2] for p in inliers))
        if best is None or key > best[0]:
            best = (key, inliers)
    if best is None:
        return None
    (n, spread, _), inliers = best
    if n < min_numerals or spread < min_spread:
        return None
    def line_at(target: float) -> float:
        """Fitted value at q = target, from the numerals NOT at that end (a misread end cannot pull its own fit)."""
        rest = [p for p in inliers if abs(p[0] - target) >= ends_window]
        use = rest if len(rest) >= 2 and max(p[0] for p in rest) - min(p[0] for p in rest) >= 0.2 else inliers
        a, b = np.linalg.lstsq(np.array([[1.0, q] for q, _, _ in use]), np.array([v for _, v, _ in use]), rcond=None)[0]
        return float(a + b * target)

    span = abs(line_at(1.0) - line_at(0.0))
    lo, lo_read = choose_end(line_at(0.0), [v for q, v, _ in inliers if abs(q) < ends_window], span)
    hi, hi_read = choose_end(line_at(1.0), [v for q, v, _ in inliers if abs(q - 1) < ends_window], span)
    if hi <= lo:
        return None
    return ScaleRange(lo=lo, hi=hi, read_ends=lo_read and hi_read, numerals=sorted(inliers))


# Gauge ranges come from short preferred-number series (EN 837 bar: 1, 1.6, 2.5, 4, 6, 10, 16 ...; psi: 15, 30,
# 60, 100, 160, 200, 300 ...), with 0 or -1 at the low end. Used to tidy an end value that was not read directly.
STANDARD_MANTISSAS = (1.0, 1.5, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)


def standard_value(x: float, span: float, tol: float = 0.05) -> float | None:
    """Nearest preferred-number scale end to x (0 included), if within tol * span; else None."""
    if abs(x) <= tol * span:
        return 0.0
    k = int(np.floor(np.log10(abs(x))))
    cands = [np.sign(x) * m * 10.0 ** e for e in (k - 1, k, k + 1) for m in STANDARD_MANTISSAS]
    best = min(cands, key=lambda c: abs(c - x))
    return float(np.round(best, 9)) if abs(best - x) <= tol * span else None


def choose_end(fitted: float, read: list[float], span: float, tie: float = 0.015) -> tuple[float, bool]:
    """End value of the scale from the numerals read at that end and a standard range value near the fit.

    Candidates within `tie` x span of the best fit are treated as equally good (numeral positions put the fit
    ~1 % of the span off), and then 0 wins, then a standard range value, then a numeral read: a tick misread as
    "1" next to the real "0" loses. Without candidates, the fit rounded to 3 significant figures.
    Returns (value, True if a numeral with that value was actually read).
    """
    std = standard_value(fitted, span)
    cands = {v: abs(v - fitted) for v in read}
    if std is not None:
        cands.setdefault(std, abs(std - fitted))
    if not cands:
        return float(f"{fitted:.3g}") + 0.0, False
    best = min(cands.values())
    close = [v for v, d in cands.items() if d <= best + tie * span]
    is_std = lambda v: v == 0 or standard_value(v, span, tol=1e-9) is not None
    value = min(close, key=lambda v: (v != 0, not is_std(v), not any(abs(r - v) < 1e-9 for r in read), cands[v]))
    return value + 0.0, any(abs(r - value) < 1e-9 for r in read)                  # + 0.0 turns -0.0 into 0.0


class ScaleReader:
    """Callable: reader(photo_rgb, infer_result, strategy) -> ScaleRange | None. RapidOCR is loaded lazily."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._engine = None

    def _ocr(self, img: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
        if self._engine is None:
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
        r = self._engine(img)
        if r.boxes is None:
            return []
        return [(np.asarray(b, float), str(t), float(c)) for b, t, c in zip(r.boxes, r.txts, r.scores)]

    def numerals(self, view: np.ndarray, k: np.ndarray, ring: float) -> list[tuple[float, float, float]]:
        """(q, value, conf) for numbers found near the scale keypoints k (upright view px)."""
        c = self.cfg
        found = []
        for i in range(2, 7):
            d = (k[i] - k[0]) / np.linalg.norm(k[i] - k[0])
            for rho in c["patch_radii"]:
                centre = k[0] + d * rho * ring
                half = c["patch_half"] * ring
                x0, y0 = int(max(centre[0] - half, 0)), int(max(centre[1] - half, 0))
                patch = view[y0:int(centre[1] + half), x0:int(centre[0] + half)]
                if patch.size == 0:
                    continue
                big = cv2.resize(patch, None, fx=c["upscale"], fy=c["upscale"], interpolation=cv2.INTER_CUBIC)
                for box, text, conf in self._ocr(big):
                    v = parse_number(text)
                    if v is not None and conf >= c["min_conf"]:
                        found.append((box.mean(axis=0) / c["upscale"] + [x0, y0], v, conf))
        kept = []                                    # one entry per numeral (patches overlap): most confident wins
        for pos, v, conf in sorted(found, key=lambda x: -x[2]):
            if all(np.linalg.norm(pos - p) > 0.06 * ring for p, _, _ in kept):
                kept.append((pos, v, conf))
        out = []
        for pos, v, conf in kept:
            if 0.35 <= np.linalg.norm(pos - k[0]) / ring <= 1.5:
                q = needle_fraction(k[0], pos, k[2], k[6])
                if -0.12 <= q <= 1.12:
                    out.append((q, v, conf))
        return out

    def __call__(self, image: np.ndarray, res: dict, strategy: str | None) -> ScaleRange | None:
        size = int(self.cfg["view_size"])
        U = upright_matrix(res, strategy, size)
        if U is None:
            return None
        G = U @ res["crop_matrix"]                                                    # photo px -> upright px
        view = cv2.warpPerspective(image, G, (size, size), flags=cv2.INTER_CUBIC)
        k = apply_h(U, res["keypoints"])
        ring = float(np.mean(np.linalg.norm(k[2:] - k[0], axis=1)))
        c = self.cfg
        return fit_range(self.numerals(view, k, ring), c["min_numerals"], c["tol"], c["ends_window"], c["min_spread"])
