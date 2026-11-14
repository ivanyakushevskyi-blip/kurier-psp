"""Порівняння для Івана (15.11.2026): що є в OSM, що на знімку, як намальовано зараз і новим кодом.
  python3 tools/osmview.py x,y [W H] out.png      (координати — пікселі карти гри)"""
import sys, os, subprocess, math
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import citydraw as CD

LAND = {'park': (170, 220, 160), 'grass': (190, 230, 170), 'wood': (140, 200, 130), 'pitch': (120, 200, 140), 'playground': (210, 240, 200),
        'parking': (220, 220, 245), 'water': (170, 200, 240), 'residential': (232, 228, 224), 'commercial': (245, 225, 225),
        'retail': (245, 220, 215), 'industrial': (230, 215, 235), 'pedestrian': (235, 230, 215), 'garages': (225, 220, 210)}
RCOL = {'primary': (240, 160, 100), 'secondary': (240, 200, 110), 'tertiary': (250, 240, 150), 'service': (255, 255, 255)}


def osm(x0, y0, W, H):
    D = CD.data(); K = CD.K; F = 1.0 / CD.MPP
    img = np.full((H, W, 3), 246, np.uint8)
    P = lambda pts: np.round(np.array([(p[0] * K - x0, p[1] * K - y0) for p in pts])).astype(np.int32)
    vis = lambda w: not (w['bb'][2] * K < x0 - 50 or w['bb'][0] * K > x0 + W + 50 or w['bb'][3] * K < y0 - 50 or w['bb'][1] * K > y0 + H + 50)
    ws = [w for w in D['w'] if vis(w)]
    areas = []
    for w in ws:
        if not w['closed']: continue
        t = w['t']; k = None
        for key in ('leisure', 'landuse', 'amenity', 'natural', 'highway'):
            v = t.get(key)
            if v in LAND: k = v; break
            if v in ('forest', 'scrub'): k = 'wood'; break
            if v in ('meadow', 'flowerbed', 'village_green', 'grassland'): k = 'grass'; break
        if k: areas.append((cv2.contourArea(P(w['pts']).astype(np.float32)), k, w))
    for _, k, w in sorted(areas, key=lambda a: -a[0]):
        cv2.fillPoly(img, [P(w['pts'])], LAND[k][::-1]); cv2.polylines(img, [P(w['pts'])], True, tuple(int(c * 0.8) for c in LAND[k][::-1]), 1)
    for b in D['B'] + D['SAT']:
        q = P(b['q'])
        if q[:, 0].max() < 0 or q[:, 0].min() > W or q[:, 1].max() < 0 or q[:, 1].min() > H: continue
        src = b.get('src', '')
        col = (200, 196, 214) if 'ms' in str(src).lower() or b in D['SAT'] else (190, 180, 170)
        cv2.fillPoly(img, [q], col); cv2.polylines(img, [q], True, (110, 100, 95), 1)
    for w in ws:
        t = w['t']; h = t.get('highway')
        if w['closed'] and t.get('area') == 'yes': continue
        if h in CD.ROAD_W:
            q = P(w['pts']); th = max(2, int(CD.ROAD_W[h] * F))
            cv2.polylines(img, [q], False, (90, 90, 90), th + 2); cv2.polylines(img, [q], False, RCOL.get(h, (255, 255, 255))[::-1], th)
    for w in ws:
        h = w['t'].get('highway')
        if h in ('footway', 'path', 'pedestrian', 'steps', 'cycleway'):
            q = P(w['pts']).astype(np.float32)
            for i in range(len(q) - 1):
                a, b = q[i], q[i + 1]; L = float(np.hypot(*(b - a)))
                for s in np.arange(0, L, 8):
                    e = min(L, s + 5); cv2.line(img, tuple(np.round(a + (b - a) * s / max(L, 1e-6)).astype(int)), tuple(np.round(a + (b - a) * e / max(L, 1e-6)).astype(int)), (60, 60, 220), 2)
        if w['t'].get('railway') in ('rail', 'tram'):
            cv2.polylines(img, [P(w['pts'])], False, (80, 80, 80), 3)
    for w in ws:                                                         # назви вулиць
        n = w['t'].get('name'); h = w['t'].get('highway')
        if not n or h not in CD.ROAD_W or h == 'service': continue
        q = P(w['pts']); m = q[len(q) // 2]
        if 10 < m[0] < W - 60 and 10 < m[1] < H - 10:
            pass   # кирилиця в cv2 не малюється — підписи робить зовнішній складальник
    return img


def sat(x0, y0, W, H):
    K = CD.K; s = np.load('build/sat_map.npy', mmap_mode='r')
    a = np.array(s[int(y0 / K):int((y0 + H) / K) + 1, int(x0 / K):int((x0 + W) / K) + 1])
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)[..., ::-1]


if __name__ == '__main__':
    x, y = map(int, sys.argv[1].split(',')); W, H = int(sys.argv[2]), int(sys.argv[3]); out = sys.argv[4]
    x0, y0 = x - W // 2, y - H // 2
    o = osm(x0, y0, W, H); s_ = sat(x0, y0, W, H)
    np.save(out + '.osm.npy', o); np.save(out + '.sat.npy', s_)
