"""Photometric nuisances applied after compositing (PLAN §3 step 3).

Frame: output image pixels, x right, y down. Images are float32 RGB in [0, 255], shape (H, W, 3).
Each factor draws only from its own generator (see `gen.generate.stream`), and always draws its
on/off decision first, so switching one factor off never changes what the others do.
"""

from __future__ import annotations

import cv2
import numpy as np

ORDER = ("glare", "shadow", "low_light", "blur", "camera")


def _u(rng, lo_hi) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def glare(img, rng, cfg, face_mask, face_poly):
    """Additive elliptical Gaussian highlight (or thin streak) clipped to the face (glass) region."""
    H, W = img.shape[:2]
    lo, hi = face_poly.min(0), face_poly.max(0)
    diam = float(np.max(hi - lo))
    centre = rng.uniform(lo, hi)
    sigma = _u(rng, cfg["size_frac"]) * diam
    streak = bool(rng.uniform() < cfg["streak_prob"])
    aspect = _u(rng, cfg["streak_aspect"] if streak else cfg["aspect"])
    a = rng.uniform(0, np.pi)
    peak = _u(rng, cfg["peak"]) * 255
    tint = rng.uniform(0.92, 1.0, 3)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - centre[0], yy - centre[1]
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    s_u, s_v = sigma, sigma / aspect
    g = np.exp(-0.5 * ((u / s_u) ** 2 + (v / s_v) ** 2))
    out = img + (peak * g * face_mask)[..., None] * tint
    return np.clip(out, 0, 255), {"streak": streak, "peak": peak / 255}


def shadow(img, rng, cfg):
    """Darken a random soft-edged polygon by 20-60 %."""
    H, W = img.shape[:2]
    n = int(rng.integers(cfg["vertices"][0], cfg["vertices"][1] + 1))
    centre = rng.uniform([0, 0], [W, H])
    radius = rng.uniform(0.3, 1.0, n) * max(W, H) * 0.6
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    pts = centre + np.column_stack([np.cos(ang), np.sin(ang)]) * radius[:, None]
    mask = np.zeros((H, W), np.float32)
    cv2.fillPoly(mask, [np.rint(pts * 16).astype(np.int32)], 1.0, lineType=cv2.LINE_AA, shift=4)
    soft = _u(rng, cfg["softness_px"])
    mask = cv2.GaussianBlur(mask, (0, 0), soft)
    strength = _u(rng, cfg["strength"])
    return img * (1 - strength * mask)[..., None], {"strength": strength}


def low_light(img, rng, cfg):
    """Gamma + gain darkening, then Poisson shot noise and Gaussian read noise."""
    gamma, gain = _u(rng, cfg["gamma"]), _u(rng, cfg["gain"])
    full_well, read = _u(rng, cfg["full_well"]), _u(rng, cfg["read_noise"])
    x = gain * (np.clip(img, 0, 255) / 255.0) ** gamma
    electrons = rng.poisson(x * full_well).astype(np.float32)
    out = electrons / full_well * 255 + rng.normal(0, read, img.shape).astype(np.float32)
    return np.clip(out, 0, 255), {"gamma": gamma, "gain": gain}


def blur(img, rng, cfg):
    """Linear motion blur or Gaussian defocus."""
    if rng.uniform() < cfg["motion_prob"]:
        length = int(rng.integers(cfg["motion_len_px"][0], cfg["motion_len_px"][1] + 1))
        a = rng.uniform(0, 180)
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1.0
        R = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), a, 1.0)
        k = cv2.warpAffine(k, R, (length, length))
        k /= max(k.sum(), 1e-6)
        return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT), {"kind": "motion", "len": length}
    sigma = _u(rng, cfg["defocus_sigma_px"])
    return cv2.GaussianBlur(img, (0, 0), sigma), {"kind": "defocus", "sigma": sigma}


def camera(img, rng, cfg):
    """Colour jitter, white balance, vignetting; picks the JPEG quality used at encode time."""
    H, W = img.shape[:2]
    b, c, s = _u(rng, cfg["brightness"]), _u(rng, cfg["contrast"]), _u(rng, cfg["saturation"])
    wb = rng.uniform(cfg["white_balance"][0], cfg["white_balance"][1], 3).astype(np.float32)
    vig = _u(rng, cfg["vignette"])
    q = int(rng.integers(cfg["jpeg_quality"][0], cfg["jpeg_quality"][1] + 1))

    out = img * b
    out = (out - out.mean()) * c + out.mean()
    grey = out @ np.array([0.299, 0.587, 0.114], np.float32)
    out = grey[..., None] + (out - grey[..., None]) * s
    out = out * wb
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    out = out * (1 - vig * r2)[..., None]
    return np.clip(out, 0, 255), {"jpeg_quality": q}


def apply_nuisances(img: np.ndarray, cfg: dict, streams: dict, face_mask: np.ndarray,
                    face_poly: np.ndarray, default_quality: int) -> tuple[np.ndarray, dict]:
    """Apply enabled nuisances in ORDER. streams: name -> np.random.Generator (one per factor).

    Returns (float32 image, record dict {glare, shadow, low_light, blur, jpeg_quality, params}).
    """
    rec = {"glare": False, "shadow": False, "low_light": False, "blur": None,
           "jpeg_quality": int(default_quality), "params": {}}
    out = img.astype(np.float32)
    for name in ORDER:
        c, rng = cfg[name], streams[name]
        fire = name == "camera" or rng.uniform() < c["p"]   # decision drawn even when disabled
        if not (c["enabled"] and fire):
            continue
        if name == "glare":
            out, p = glare(out, rng, c, face_mask, face_poly)
            rec["glare"] = True
        elif name == "shadow":
            out, p = shadow(out, rng, c)
            rec["shadow"] = True
        elif name == "low_light":
            out, p = low_light(out, rng, c)
            rec["low_light"] = True
        elif name == "blur":
            out, p = blur(out, rng, c)
            rec["blur"] = p["kind"]
        else:
            out, p = camera(out, rng, c)
            rec["jpeg_quality"] = p["jpeg_quality"]
        rec["params"][name] = p
    return out, rec
