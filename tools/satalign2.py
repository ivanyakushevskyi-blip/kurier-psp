"""Підкладка v2 (18.10.2026) для розширеної карти.
Джерела: знімки Івана ref/khn/{khm,khm2,ozerna}.jpg (≈0,785 м/пікс, Google Earth) + стара детальна
підкладка build_v1/sat_map.npy (0,32 м/пікс, уже прив'язана до OSM) — вона має пріоритет там, де є.
1) початкові афінні «піксель -> метри»: data/khn_init.json (SIFT до старої підкладки; ozerna — через khm);
2) поле поправок за MI з OSM: грубий прохід на сітці 2 м, тонкий на 1 м;
3) запис build/sat_map.npy (0,32 м) у сітці, вирівняній зі старою (цілий зсув пікселів), + маска деталізації,
   build/sat_map.json (x0, y0, W, H, mpp, off — зсув старої сітки в нову, пікс)."""
import json, sys, time, os
import numpy as np, cv2
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import satmosaic as S, mireg as MI, satalign as SA

MPP = 0.32
NAMES = ['khm', 'khm2', 'ozerna']
OLD = 'build_v1'


SOUTH = ['ternopilska', 'mazepy', 'pilotska', 'khnu']      # 24.10.2026: південь за колією, 0,336 м/пікс (хну 0,164)


def images():
    """(знімок, афінне «піксель -> метри», вага). Детальніші знімки мають більшу вагу там, де знімки перекриваються;
    кольори південних підганяються під khm/khm2/ozerna за спільною ділянкою (без «латок» на стиках)."""
    init = json.load(open('data/khn_init.json'))
    base = [(cv2.imread('ref/khn/%s.jpg' % n), np.array(init[n][1]), 1.0) for n in NAMES]
    out = list(base)
    if os.path.exists('data/khn_south_init.json'):
        si = json.load(open('data/khn_south_init.json'))
        for n in SOUTH:
            if n not in si: continue
            img = cv2.imread('ref/khn_south/%s.jpg' % n); M = np.array(si[n][1])
            res = float(np.hypot(M[0, 0], M[1, 0]))
            if res < MPP * 0.8: img = cv2.GaussianBlur(img, (0, 0), 0.45 * MPP / res)     # проти мерехтіння при зменшенні
            img = match_colors(img, M, out)
            out.append((img, M, min(10.0, (0.785 / res) ** 2)))
            print('  south', n, 'res %.3f m/px' % res, flush=True)
    return out


def match_colors(img, M, base):
    """Гістограмна відповідність кольорів знімка до основних знімків у спільній ділянці."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[h // 20:h - h // 20:6, w // 20:w - w // 20:6].astype(np.float32)
    X = M[0, 0] * xx + M[0, 1] * yy + M[0, 2]; Y = M[1, 0] * xx + M[1, 1] * yy + M[1, 2]
    ref = mosaic(base, X, Y).reshape(-1, 3)
    src = img[yy.astype(int), xx.astype(int)].reshape(-1, 3)
    ok = ref.sum(1) > 0
    if ok.sum() < 5000: print('    мало перекриття для кольорів', ok.sum()); return img
    qs = np.linspace(0, 100, 101)
    lut = np.stack([np.interp(np.arange(256), np.percentile(src[ok, k], qs), np.percentile(ref[ok, k], qs)) for k in range(3)]).clip(0, 255).astype(np.uint8)
    return np.stack([lut[k][img[..., k]] for k in range(3)], -1)


def feather(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
    return np.clip(d / 150.0, 0, 1) ** 2


def mosaic(imgs, X, Y):
    """Колір у точках (X, Y) метрів (масиви float32) зі знімків Івана з м'якими швами."""
    acc = np.zeros(X.shape + (3,), np.float32); ws = np.zeros(X.shape, np.float32)
    for it in imgs:
        img, M = it[0], it[1]; wt = it[2] if len(it) > 2 else 1.0
        Mi = np.linalg.inv(M)
        u = (Mi[0, 0] * X + Mi[0, 1] * Y + Mi[0, 2]).astype(np.float32)
        v = (Mi[1, 0] * X + Mi[1, 1] * Y + Mi[1, 2]).astype(np.float32)
        h, w = img.shape[:2]
        if not hasattr(mosaic, 'fw'): mosaic.fw = {}
        key = id(img)
        if key not in mosaic.fw: mosaic.fw[key] = feather(h, w)
        a = cv2.remap(img, u, v, cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
        b = cv2.remap(mosaic.fw[key], u, v, cv2.INTER_LINEAR, borderValue=0)
        b = b * wt
        acc += a * b[..., None]; ws += b
    ok = ws > 1e-3
    out = np.zeros(X.shape + (3,), np.uint8)
    out[ok] = (acc[ok] / ws[ok, None]).clip(0, 255).astype(np.uint8)
    return out


def grid_img(imgs, x0, y0, W, H, s, field=None):
    out = np.zeros((H, W, 3), np.uint8)
    for r0 in range(0, H, 512):
        hh = min(512, H - r0)
        yy, xx = np.mgrid[r0:r0 + hh, 0:W].astype(np.float32)
        X = x0 + xx * s; Y = y0 + yy * s
        if field is not None:
            dX, dY = field(X, Y); X = X + dX; Y = Y + dY
        out[r0:r0 + hh] = mosaic(imgs, X, Y)
    return out


def shift_field(L, C, s, gx0, gy0, win, step, srch, st):
    H, W = L.shape
    pts, sh, q = [], [], []
    t = time.time()
    for cy in range(win // 2 + srch, H - win // 2 - srch, step):
        for cx in range(win // 2 + srch, W - win // 2 - srch, step):
            T = L[cy - win // 2: cy + win // 2, cx - win // 2: cx + win // 2]
            if (T == 1).mean() < 0.04: continue
            if (C[cy - win // 2: cy + win // 2, cx - win // 2: cx + win // 2] == 0).mean() > 0.02: continue
            v, dx, dy = MI.best_shift(L, C, cx, cy, win, srch, st)
            best = (v, dx, dy)
            for ddy in range(-st + 1, st):
                for ddx in range(-st + 1, st):
                    Sx = C[cy - win // 2 + dy + ddy: cy + win // 2 + dy + ddy, cx - win // 2 + dx + ddx: cx + win // 2 + dx + ddx]
                    if Sx.shape == T.shape:
                        vv = MI.mi(T, Sx)
                        if vv > best[0]: best = (vv, dx + ddx, dy + ddy)
            v0 = MI.mi(T, C[cy - win // 2: cy + win // 2, cx - win // 2: cx + win // 2])
            pts.append((gx0 + cx * s, gy0 + cy * s)); sh.append((best[1] * s, best[2] * s)); q.append((best[0], v0, srch * s))
    print('  windows %d in %.0fs' % (len(pts), time.time() - t), flush=True)
    return np.array(pts), np.array(sh), np.array(q)


def clean(P, Sh, q, rad, tol, minq):
    ok = (q[:, 0] > minq) & (np.abs(Sh).max(1) < q[:, 2] - 1e-6)
    P, Sh = P[ok], Sh[ok]
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        d = np.hypot(*(P - P[i]).T); nb = (d < rad) & (d > 0)
        if nb.sum() >= 3 and np.hypot(*(Sh[i] - np.median(Sh[nb], 0))) > tol: keep[i] = False
    print('  reliable %d / %d, median |shift| %.1f m' % (keep.sum(), len(ok), np.median(np.hypot(*Sh[keep].T))), flush=True)
    return P[keep], Sh[keep]


def rbf_field(rbf, cs=24):
    def f(X, Y):
        h, w = X.shape
        xs, ys = X[::cs, ::cs], Y[::cs, ::cs]
        r = rbf(np.stack([xs.ravel(), ys.ravel()], 1)).reshape(xs.shape + (2,))
        return [cv2.resize(r[..., k].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) for k in range(2)]
    return f


def fields(imgs, ways, nx0, ny0, W, H):
    # ---- 1) грубий прохід, 2 м
    s = 2.0; pad = 150
    gx0, gy0 = nx0 - pad, ny0 - pad
    gw, gh = int((W * MPP + 2 * pad) / s), int((H * MPP + 2 * pad) / s)
    m2 = grid_img(imgs, gx0, gy0, gw, gh, s)
    L = MI.osm_labels(ways, gx0, gy0, gw, gh, s); C = MI.sat_classes(m2)
    P, Sh, q = shift_field(L, C, s, gx0, gy0, win=120, step=45, srch=20, st=2)
    P1, S1 = clean(P, Sh, q, 500, 12, 0.035)
    rbf1 = RBFInterpolator(P1, S1, kernel='thin_plate_spline', smoothing=2000.0)
    f1 = rbf_field(rbf1)
    # ---- 2) тонкий прохід, 1 м
    s = 1.0
    gw, gh = int((W * MPP + 2 * pad) / s), int((H * MPP + 2 * pad) / s)
    m1 = grid_img(imgs, gx0, gy0, gw, gh, s, f1)
    L = MI.osm_labels(ways, gx0, gy0, gw, gh, s); C = MI.sat_classes(m1, k=14)
    del m1, m2
    P, Sh, q = shift_field(L, C, s, gx0, gy0, win=180, step=80, srch=12, st=2)
    P2, S2 = clean(P, Sh, q, 280, 6, 0.05)
    rbf2 = RBFInterpolator(P2, S2, kernel='thin_plate_spline', smoothing=800.0)
    f2 = rbf_field(rbf2)
    json.dump(dict(P1=P1.tolist(), S1=S1.tolist(), P2=P2.tolist(), S2=S2.tolist()), open('build/satalign2.json', 'w'))

    return f1, f2


def main():
    imgs = images()
    ways = S.load_osm('data/osm_v3')
    om = json.load(open(OLD + '/sat_map.json'))
    x0, y0, x1, y1 = SA.map_grid()
    # нова сітка: ціле число пікселів 0,32 м від старої
    ox = int(np.ceil((om['x0'] - x0) / MPP)); oy = int(np.ceil((om['y0'] - y0) / MPP))
    nx0 = om['x0'] - ox * MPP; ny0 = om['y0'] - oy * MPP
    W = int(np.ceil((x1 - nx0) / MPP)); H = int(np.ceil((y1 - ny0) / MPP))
    print('grid', W, H, 'old offset', ox, oy, flush=True)
    if os.path.exists('build/satalign2.json') and os.environ.get('REALIGN') != '1':
        j = json.load(open('build/satalign2.json'))
        P1, S1, P2, S2 = (np.array(j[k]) for k in ('P1', 'S1', 'P2', 'S2'))
        f1 = rbf_field(RBFInterpolator(P1, S1, kernel='thin_plate_spline', smoothing=2000.0))
        f2 = rbf_field(RBFInterpolator(P2, S2, kernel='thin_plate_spline', smoothing=800.0))
    else:
        f1, f2 = fields(imgs, ways, nx0, ny0, W, H)

    def total(X, Y):
        a, b = f2(X, Y); X2, Y2 = X + a, Y + b
        c, d = f1(X2, Y2)
        return a + c, b + d
    # ---- 3) підсумкова підкладка 0,32 м
    if os.path.exists('build/sat_map.npy'): os.remove('build/sat_map.npy')     # не писати в спільний inode (жорсткі посилання)
    out = np.lib.format.open_memmap('build/sat_map.npy', 'w+', np.uint8, (H, W, 3))
    det = np.zeros((H, W), bool)
    old = np.load(OLD + '/sat_map.npy', mmap_mode='r'); odet = np.load(OLD + '/sat_detail_mask.npy')
    oh, ow = odet.shape
    # маска старої деталізації в новій сітці + м'який перехід 25 м
    wold = np.zeros((H, W), np.float32)
    wold[oy:oy + oh, ox:ox + ow] = ndi.binary_erosion(odet, iterations=20)
    wold = cv2.GaussianBlur(wold, (0, 0), 25)
    wold[oy:oy + oh, ox:ox + ow] *= odet
    det[oy:oy + oh, ox:ox + ow] = odet
    # кольори знімків Івана підганяємо під стару підкладку (гістограми в зоні перекриття)
    lut = None
    for r0 in range(0, H, 400):
        hh = min(400, H - r0)
        yy, xx = np.mgrid[r0:r0 + hh, 0:W].astype(np.float32)
        X = nx0 + xx * MPP; Y = ny0 + yy * MPP
        dX, dY = total(X, Y)
        a = mosaic(imgs, X + dX, Y + dY)
        if lut is None:
            # гістограмна відповідність за вибіркою в центрі старої зони
            sy0, sx0 = oy + oh // 2 - 1500, ox + ow // 2 - 1500
            yy2, xx2 = np.mgrid[sy0:sy0 + 3000:3, sx0:sx0 + 3000:3].astype(np.float32)
            X2 = nx0 + xx2 * MPP; Y2 = ny0 + yy2 * MPP
            d2 = total(X2, Y2)
            src = mosaic(imgs, X2 + d2[0], Y2 + d2[1]).reshape(-1, 3)
            ref = np.asarray(old[sy0 - oy:sy0 - oy + 3000:3, sx0 - ox:sx0 - ox + 3000:3]).reshape(-1, 3)
            ok = (src.sum(1) > 0) & (ref.sum(1) > 0)
            lut = np.zeros((3, 256), np.uint8)
            for k in range(3):
                qs = np.linspace(0, 100, 101)
                ps, pr = np.percentile(src[ok, k], qs), np.percentile(ref[ok, k], qs)
                lut[k] = np.interp(np.arange(256), ps, pr).clip(0, 255).astype(np.uint8)
        a = np.stack([lut[k][a[..., k]] for k in range(3)], -1)
        # стара детальна підкладка
        ra, rb = max(r0, oy), min(r0 + hh, oy + oh)
        if rb > ra:
            w = wold[r0:r0 + hh, ox:ox + ow][ra - r0:rb - r0, :, None]
            o = np.asarray(old[ra - oy:rb - oy]).astype(np.float32)
            seg = a[ra - r0:rb - r0, ox:ox + ow].astype(np.float32)
            a[ra - r0:rb - r0, ox:ox + ow] = (o * w + seg * (1 - w)).astype(np.uint8)
        out[r0:r0 + hh] = a
        print('  rows', r0, flush=True)
    out.flush()
    np.save('build/sat_detail_mask.npy', det)
    json.dump(dict(x0=nx0, y0=ny0, W=W, H=H, mpp=MPP, off=[ox, oy]), open('build/sat_map.json', 'w'))
    print('sat_map', H, W)


if __name__ == '__main__':
    main()
