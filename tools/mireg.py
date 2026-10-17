"""Прив'язка знімка до OSM за взаємною інформацією (MI) між класами OSM і кольорами знімка."""
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

RW = {'primary': 14, 'secondary': 12, 'tertiary': 10, 'residential': 7, 'unclassified': 7, 'trunk': 14, 'service': 4, 'living_street': 5, 'pedestrian': 8}


def osm_labels(ways, x0, y0, W, H, s):
    im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
    def px(p): return [((x - x0) / s, (y - y0) / s) for x, y in p]
    for p, t in ways:
        if len(p) < 3: continue
        if t.get('leisure') in ('park', 'garden', 'pitch') or t.get('landuse') in ('grass', 'forest', 'meadow') or t.get('natural') in ('wood', 'scrub', 'grassland'):
            d.polygon(px(p), fill=4)
    for p, t in ways:
        if len(p) >= 3 and t.get('natural') == 'water': d.polygon(px(p), fill=3)
        if t.get('waterway') in ('river', 'stream', 'canal') and len(p) >= 2: d.line(px(p), fill=3, width=max(1, int(12 / s)))
    for p, t in ways:
        if t.get('highway') in RW and len(p) >= 2: d.line(px(p), fill=2, width=max(1, int(RW[t['highway']] / s)))
        if t.get('railway') == 'rail' and len(p) >= 2: d.line(px(p), fill=5, width=max(1, int(6 / s)))
    for p, t in ways:
        if 'building' in t and len(p) >= 3: d.polygon(px(p), fill=1)
    return np.array(im)


def sat_classes(img_bgr, k=12, seed=0):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    valid = img_bgr.reshape(-1, 3).sum(1) > 0
    sample = lab[valid][np.random.default_rng(seed).choice(valid.sum(), min(200000, valid.sum()), replace=False)]
    _, _, C = cv2.kmeans(sample, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    d = ((lab[:, None, :] - C[None]) ** 2).sum(-1)
    cl = d.argmin(1).astype(np.uint8) + 1
    cl[~valid] = 0
    return cl.reshape(img_bgr.shape[:2])


def mi(a, b, na=6, nb=16):
    m = (b > 0)
    h = np.bincount(a[m].astype(np.int32) * nb + b[m].astype(np.int32), minlength=na * nb).reshape(na, nb).astype(np.float64)
    h /= h.sum() + 1e-12
    pa = h.sum(1, keepdims=True); pb = h.sum(0, keepdims=True)
    nz = h > 0
    return float((h[nz] * np.log(h[nz] / (pa @ pb)[nz])).sum())


def best_shift(L, C, cx, cy, win, srch, step):
    T = L[cy - win // 2: cy + win // 2, cx - win // 2: cx + win // 2]
    best = (-1, 0, 0)
    for dy in range(-srch, srch + 1, step):
        for dx in range(-srch, srch + 1, step):
            Sx = C[cy - win // 2 + dy: cy + win // 2 + dy, cx - win // 2 + dx: cx + win // 2 + dx]
            if Sx.shape != T.shape: continue
            v = mi(T, Sx)
            if v > best[0]: best = (v, dx, dy)
    return best
