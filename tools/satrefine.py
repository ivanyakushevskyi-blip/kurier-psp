"""Другий, дрібніший прохід прив'язки: вікна 200 м на сітці 1 м, пошук ±24 м навколо першого наближення."""
import json, sys, time
import numpy as np, cv2
from scipy.interpolate import RBFInterpolator
sys.path.insert(0, __import__('os').path.dirname(__file__))
import satmosaic as S, mireg as M

meta = json.load(open('build/sat_map.json')); sat = np.load('build/sat_map.npy')
x0, y0, mpp = meta['x0'], meta['y0'], meta['mpp']
ways = S.load_osm('data/osm_v2')
s = 1.0; f = s / mpp
small = cv2.resize(sat, (int(sat.shape[1] / f), int(sat.shape[0] / f)), interpolation=cv2.INTER_AREA)
H, W = small.shape[:2]
L = M.osm_labels(ways, x0, y0, W, H, s)
C = M.sat_classes(small, k=14)
win, step, srch = 200, 70, 24
pts, sh, q = [], [], []
t = time.time()
for cy in range(win // 2 + srch, H - win // 2 - srch, step):
    for cx in range(win // 2 + srch, W - win // 2 - srch, step):
        T = L[cy - win // 2: cy + win // 2, cx - win // 2: cx + win // 2]
        if (T == 1).mean() < 0.05: continue
        if (C[cy - win // 2: cy + win // 2, cx - win // 2: cx + win // 2] == 0).mean() > 0.02: continue
        v, dx, dy = M.best_shift(L, C, cx, cy, win, srch, 2)
        best = (v, dx, dy)
        for ddy in (-1, 0, 1):
            for ddx in (-1, 0, 1):
                Sx = C[cy - win // 2 + dy + ddy: cy + win // 2 + dy + ddy, cx - win // 2 + dx + ddx: cx + win // 2 + dx + ddx]
                if Sx.shape == T.shape:
                    vv = M.mi(T, Sx)
                    if vv > best[0]: best = (vv, dx + ddx, dy + ddy)
        v0 = M.mi(T, C[cy - win // 2: cy + win // 2, cx - win // 2: cx + win // 2])
        pts.append((cx * s, cy * s)); sh.append((best[1] * s, best[2] * s)); q.append((best[0], v0))
pts, sh, q = np.array(pts), np.array(sh), np.array(q)
print('windows', len(pts), '%.0fs' % (time.time() - t))
ok = (q[:, 0] > 0.05) & (np.abs(sh).max(1) < srch - 1)
P, Sh = pts[ok], sh[ok]
keep = np.ones(len(P), bool)
for i in range(len(P)):
    d = np.hypot(*(P - P[i]).T); nb = (d < 260) & (d > 0)
    if nb.sum() >= 3 and np.hypot(*(Sh[i] - np.median(Sh[nb], 0))) > 6: keep[i] = False
P, Sh = P[keep], Sh[keep]
print('reliable', len(P), 'median |shift| %.1f m, 90%% %.1f m' % (np.median(np.hypot(*Sh.T)), np.percentile(np.hypot(*Sh.T), 90)))
rbf = RBFInterpolator(P, Sh, kernel='thin_plate_spline', smoothing=800.0)
Hh, Ww = sat.shape[:2]
out = np.zeros_like(sat)
det = np.load('build/sat_detail_mask.npy').astype(np.uint8) * 255; det2 = np.zeros_like(det)
for yy0 in range(0, Hh, 500):
    hh = min(500, Hh - yy0)
    cs = 25
    ys_, xs_ = np.mgrid[yy0:yy0 + hh:cs, 0:Ww:cs]
    r = rbf(np.stack([xs_.ravel() * mpp, ys_.ravel() * mpp], 1)).reshape(xs_.shape + (2,))
    r = np.stack([cv2.resize(r[..., k].astype(np.float32), (Ww, hh), interpolation=cv2.INTER_LINEAR) for k in range(2)], -1)
    yy, xx = np.mgrid[yy0:yy0 + hh, 0:Ww].astype(np.float32)
    mx = xx + r[..., 0] / mpp; my = yy + r[..., 1] / mpp
    out[yy0:yy0 + hh] = cv2.remap(sat, mx, my, cv2.INTER_LINEAR, borderValue=0)
    det2[yy0:yy0 + hh] = cv2.remap(det, mx, my, cv2.INTER_NEAREST, borderValue=0)
np.save('build/sat_map.npy', out); np.save('build/sat_detail_mask.npy', det2 > 128)
json.dump(dict(P=P.tolist(), Sh=Sh.tolist()), open('build/satrefine.json', 'w'))
print('refined')
