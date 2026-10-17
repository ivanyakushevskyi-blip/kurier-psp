"""Повний конвеєр супутникової підкладки:
широкий знімок + 14 детальних тайлів -> мозаїка (наближені метри) -> поле поправок за MI з OSM ->
вирівняна підкладка в сітці карти (build/sat_map.npy, 0.32 м/піксель) + маска покриття детальними знімками."""
import glob, json, math, sys, time
import numpy as np, cv2
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator
sys.path.insert(0, __import__('os').path.dirname(__file__))
import satmosaic as S, mireg as M

LAT0, LON0 = S.LAT0, S.LON0
MPP = 0.32


def map_grid():
    B = np.load('build/bounds.npz')
    G2 = float(B['G']); lon0 = float(B['lon0']); lat1 = float(B['lat1'])
    ys, xs = np.nonzero(B['margin'])
    x0 = (lon0 - LON0) * S.MLON + xs.min() * G2; x1 = (lon0 - LON0) * S.MLON + (xs.max() + 1) * G2
    y0 = -(lat1 - LAT0) * S.MLAT + ys.min() * G2; y1 = -(lat1 - LAT0) * S.MLAT + (ys.max() + 1) * G2
    return x0, y0, x1, y1


def mosaic_with_wide(tiles, Hs, g, wide_path):
    wide = cv2.imread(wide_path)
    base = cv2.warpPerspective(wide, (g.M @ S.A0).astype(np.float64), (g.W, g.H), flags=cv2.INTER_CUBIC)
    det, cov = S.build_mosaic(tiles, Hs, g)
    w = ndi.gaussian_filter(cov.astype(np.float32), 6)
    out = (det.astype(np.float32) * w[..., None] + base.astype(np.float32) * (1 - w[..., None]))
    return out.astype(np.uint8), cov


def field(L, C, s, gx0, gy0, valid):
    win, step, srch = 250, 90, 70
    pts, sh, q = [], [], []
    H, W = L.shape
    t = time.time()
    for cy in range(win // 2 + srch, H - win // 2 - srch, step):
        for cx in range(win // 2 + srch, W - win // 2 - srch, step):
            if valid[cy, cx] == 0: continue
            T = L[cy - win // 2: cy + win // 2, cx - win // 2: cx + win // 2]
            if (T == 1).mean() < 0.04: continue
            v, dx, dy = M.best_shift(L, C, cx, cy, win, srch, 3)
            best = (v, dx, dy)
            for ddy in range(-3, 4):
                for ddx in range(-3, 4):
                    Sx = C[cy - win // 2 + dy + ddy: cy + win // 2 + dy + ddy, cx - win // 2 + dx + ddx: cx + win // 2 + dx + ddx]
                    if Sx.shape == T.shape:
                        vv = M.mi(T, Sx)
                        if vv > best[0]: best = (vv, dx + ddx, dy + ddy)
            v0 = M.mi(T, C[cy - win // 2: cy + win // 2, cx - win // 2: cx + win // 2])
            pts.append((gx0 + cx * s, gy0 + cy * s)); sh.append((best[1] * s, best[2] * s)); q.append((best[0], v0))
    print('windows %d in %.0fs' % (len(pts), time.time() - t))
    return np.array(pts), np.array(sh), np.array(q)


def clean(pts, sh, q):
    ok = (q[:, 0] > 0.035) & (q[:, 0] > q[:, 1] * 1.4)
    P, Sh = pts[ok], sh[ok]
    # прибираємо викиди: відхилення від медіани сусідів у радіусі 450 м
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        d = np.hypot(*(P - P[i]).T); nb = (d < 450) & (d > 0)
        if nb.sum() >= 3:
            med = np.median(Sh[nb], 0)
            if np.hypot(*(Sh[i] - med)) > 14: keep[i] = False
    print('reliable windows %d / %d' % (keep.sum(), len(pts)))
    return P[keep], Sh[keep]


def main():
    tiles = sorted(glob.glob('ref/sat_tiles/tile*.jpg'))
    ways = S.load_osm('data/osm_v2')
    x0, y0, x1, y1 = map_grid()
    pad = 200
    s = 2.0
    g2 = S.Grid(x0 - pad, y0 - pad, x1 + pad, y1 + pad, s)
    Hs = S.tiles_to_wide('ref/satellite_wide.jpg', tiles)
    mos2, cov2 = mosaic_with_wide(tiles, Hs, g2, 'ref/satellite_wide.jpg')
    L = M.osm_labels(ways, g2.x0, g2.y0, g2.W, g2.H, s)
    C = M.sat_classes(mos2)
    pts, sh, q = field(L, C, s, g2.x0, g2.y0, (mos2.sum(2) > 0))
    P, Sh = clean(pts, sh, q)
    rbf = RBFInterpolator(P, Sh, kernel='thin_plate_spline', smoothing=3000.0)
    pred = rbf(P); print('fit residual mean %.1f m' % np.hypot(*(pred - Sh).T).mean())
    json.dump(dict(P=P.tolist(), Sh=Sh.tolist(), Hs=[h.tolist() for h in Hs], tiles=tiles), open('build/satalign.json', 'w'))
    # фінальна підкладка в сітці карти
    gF = S.Grid(x0, y0, x1, y1, 0.5)                     # мозаїка в наближених метрах, 0.5 м
    pad2 = 120
    gM = S.Grid(x0 - pad2, y0 - pad2, x1 + pad2, y1 + pad2, 0.5)
    mos, cov = mosaic_with_wide(tiles, Hs, gM, 'ref/satellite_wide.jpg')
    W = int((x1 - x0) / MPP); H = int((y1 - y0) / MPP)
    out = np.zeros((H, W, 3), np.uint8); cmap = np.zeros((H, W), np.uint8)
    covf = cov.astype(np.uint8) * 255
    for yy0 in range(0, H, 400):
        hh = min(400, H - yy0)
        yy, xx = np.mgrid[yy0:yy0 + hh, 0:W].astype(np.float32)
        X = x0 + xx * MPP; Y = y0 + yy * MPP
        cs = 25
        r = rbf(np.stack([X[::cs, ::cs].ravel(), Y[::cs, ::cs].ravel()], 1)).reshape(X[::cs, ::cs].shape + (2,))
        r = np.stack([cv2.resize(r[..., k].astype(np.float32), (W, hh), interpolation=cv2.INTER_LINEAR) for k in range(2)], -1)
        mx = ((X + r[..., 0] - gM.x0) / gM.s).astype(np.float32); my = ((Y + r[..., 1] - gM.y0) / gM.s).astype(np.float32)
        out[yy0:yy0 + hh] = cv2.remap(mos, mx, my, cv2.INTER_LINEAR, borderValue=0)
        cmap[yy0:yy0 + hh] = cv2.remap(covf, mx, my, cv2.INTER_LINEAR, borderValue=0)
    np.save('build/sat_map.npy', out); np.save('build/sat_detail_mask.npy', cmap > 128)
    json.dump(dict(x0=x0, y0=y0, W=W, H=H, mpp=MPP), open('build/sat_map.json', 'w'))
    print('sat_map', out.shape, 'detail coverage %.2f' % (cmap > 128).mean())


if __name__ == '__main__':
    main()
