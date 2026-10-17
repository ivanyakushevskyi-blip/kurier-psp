"""Зшивання детальних супутникових знімків і прив'язка до OSM.
Етапи:
 1) кожен тайл -> широкий знімок (SIFT, гомографія) -> початкові метри (наближення A0);
 2) мозаїка в початкових метрах;
 3) локальне уточнення: у вікнах ~300 м шукаємо зсув, за якого контури будинків OSM найкраще
    збігаються з перепадами яскравості на знімку; з усіх зсувів будуємо плавне поле поправок;
 4) запис вирівняної мозаїки в сітці OSM (build/sat_aligned_*.npy) + звіт.
"""
import glob, json, math, sys, xml.etree.ElementTree as ET
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator

LAT0, LON0 = 49.422, 26.995
MLAT = 111320.0; MLON = MLAT * math.cos(math.radians(LAT0))
A0 = np.array([[1.656, 0, -850 - 880 * 1.656], [0, 1.656, -750 - 560 * 1.656], [0, 0, 1]])   # широкий px -> м (наближення)


def load_osm(osm_dir):
    nodes, ways, seen = {}, [], set()
    for f in sorted(glob.glob(osm_dir + '/t*.osm')):
        r = ET.parse(f).getroot()
        for n in r.iter('node'): nodes[n.get('id')] = (float(n.get('lat')), float(n.get('lon')))
        for w in r.iter('way'):
            if w.get('id') in seen: continue
            seen.add(w.get('id'))
            ways.append(([nd.get('ref') for nd in w.iter('nd')], {x.get('k'): x.get('v') for x in w.iter('tag')}))
    def xy(n): la, lo = nodes[n]; return ((lo - LON0) * MLON, -(la - LAT0) * MLAT)
    return [([xy(n) for n in nds if n in nodes], t) for nds, t in ways]


class Grid:
    def __init__(self, x0, y0, x1, y1, s):
        self.x0, self.y0, self.s = x0, y0, s
        self.W, self.H = int((x1 - x0) / s), int((y1 - y0) / s)
        self.M = np.array([[1 / s, 0, -x0 / s], [0, 1 / s, -y0 / s], [0, 0, 1]])   # метри -> піксель сітки

    def px(self, pts): return [((x - self.x0) / self.s, (y - self.y0) / self.s) for x, y in pts]


def osm_building_edges(ways, g):
    im = Image.new('L', (g.W, g.H), 0); d = ImageDraw.Draw(im)
    for p, t in ways:
        if 'building' in t and len(p) > 2: d.polygon(g.px(p), outline=255)
    return ndi.gaussian_filter(np.array(im).astype(np.float32) / 255, 1.2)


def tiles_to_wide(wide_path, tiles):
    wide = cv2.imread(wide_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=30000)
    kw, dw = sift.detectAndCompute(wide, None)
    fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    Hs = []
    for f in tiles:
        t0 = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        k = (2576 / t0.shape[1]) * 0.5
        t = cv2.resize(t0, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        kt, dt = sift.detectAndCompute(t, None)
        g = [a for a, b in fl.knnMatch(dt, dw, k=2) if a.distance < 0.75 * b.distance]
        src = np.float32([kt[a.queryIdx].pt for a in g]) / k
        dst = np.float32([kw[a.trainIdx].pt for a in g])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        Hs.append(H)
    return Hs


def build_mosaic(tiles, Hs, g):
    acc = np.zeros((g.H, g.W, 3), np.float32); wsum = np.zeros((g.H, g.W), np.float32)
    for f, H in zip(tiles, Hs):
        img = cv2.imread(f)
        h, w = img.shape[:2]
        # вага: максимум у центрі тайла, нуль на краях (м'які шви) + відрізаємо службові написи по краях
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        wt = np.minimum(np.minimum(xx, w - 1 - xx) / w, np.minimum(yy, h - 1 - yy) / h)
        wt = np.clip(wt * 8, 0, 1) ** 2
        wt[: int(h * 0.03)] = 0; wt[int(h * 0.97):] = 0
        M = g.M @ A0 @ H
        a = cv2.warpPerspective(img.astype(np.float32), M, (g.W, g.H), flags=cv2.INTER_AREA)
        b = cv2.warpPerspective(wt, M, (g.W, g.H), flags=cv2.INTER_LINEAR)
        acc += a * b[..., None]; wsum += b
    cov = wsum > 1e-3
    out = np.zeros_like(acc); out[cov] = acc[cov] / wsum[cov, None]
    return out.astype(np.uint8), cov


def sat_edges(img):
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gr, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gr, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy)
    return ndi.gaussian_filter(np.minimum(m / (np.percentile(m, 97) + 1e-6), 1.0), 1.0)


def local_shifts(E_osm, E_sat, cov, g, win_m=320, step_m=160, search_m=45):
    win, step, srch = int(win_m / g.s), int(step_m / g.s), int(search_m / g.s)
    pts, sh, q = [], [], []
    for cy in range(win // 2 + srch, g.H - win // 2 - srch, step):
        for cx in range(win // 2 + srch, g.W - win // 2 - srch, step):
            T = E_osm[cy - win // 2: cy + win // 2, cx - win // 2: cx + win // 2]
            if T.mean() < 0.01: continue                                  # мало будинків
            S = E_sat[cy - win // 2 - srch: cy + win // 2 + srch, cx - win // 2 - srch: cx + win // 2 + srch]
            C = cov[cy - win // 2 - srch: cy + win // 2 + srch, cx - win // 2 - srch: cx + win // 2 + srch]
            if C.mean() < 0.98: continue
            R = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(R)
            # якість піку: порівняння з другим максимумом поза околом
            R2 = R.copy(); cv2.circle(R2, loc, int(8 / g.s), -1, -1)
            second = R2.max()
            dx, dy = (loc[0] - srch) * g.s, (loc[1] - srch) * g.s
            pts.append((g.x0 + cx * g.s, g.y0 + cy * g.s)); sh.append((dx, dy)); q.append((mx, mx - second))
    return np.array(pts), np.array(sh), np.array(q)


def main(osm_dir, wide, tiles, out_dir):
    ways = load_osm(osm_dir)
    Hs = tiles_to_wide(wide, tiles)
    B = np.load('build/bounds.npz')
    # межі відмальовування (запас) у метрах
    ys, xs = np.nonzero(B['margin'])
    G2 = float(B['G']); lon0 = float(B['lon0']); lat1 = float(B['lat1'])
    bx0 = (lon0 - LON0) * MLON + xs.min() * G2; bx1 = (lon0 - LON0) * MLON + xs.max() * G2
    by0 = -(lat1 - LAT0) * MLAT + ys.min() * G2; by1 = -(lat1 - LAT0) * MLAT + ys.max() * G2
    pad = 80
    g = Grid(bx0 - pad, by0 - pad, bx1 + pad, by1 + pad, 0.5)
    print('grid', g.W, g.H)
    mos, cov = build_mosaic(tiles, Hs, g)
    E_sat = sat_edges(mos); E_osm = osm_building_edges(ways, g)
    pts, sh, q = local_shifts(E_osm, E_sat, cov, g)
    print('windows', len(pts))
    good = (q[:, 0] > 0.18) & (q[:, 1] > 0.03)
    print('good', good.sum())
    # глобальна частина: афінне за надійними вікнами (RANSAC), далі гладкий залишок
    P, S = pts[good], sh[good]
    A, inl = cv2.estimateAffine2D(P.astype(np.float32), (P - S).astype(np.float32), ransacReprojThreshold=6)
    inl = inl.ravel().astype(bool)
    resid = (np.hstack([P, np.ones((len(P), 1))]) @ A.T) - (P - S)
    print('affine inliers %d/%d, mean resid %.1f m' % (inl.sum(), len(P), np.hypot(*resid[inl].T).mean()))
    rbf = RBFInterpolator(P[inl], resid[inl], kernel='thin_plate_spline', smoothing=200.0)
    json.dump(dict(A=A.tolist(), pts=P[inl].tolist(), resid=resid[inl].tolist(), grid=[g.x0, g.y0, g.W, g.H, g.s],
                   Hs=[h.tolist() for h in Hs], tiles=tiles), open(out_dir + '/satreg.json', 'w'))
    np.save(out_dir + '/sat_mosaic_init.npy', mos)
    return A, rbf, P, S, inl, g


def aligned(mos, g, A, rbf, out_s, ox0, oy0, W, H, chunk=512):
    """Вирівняна мозаїка в сітці OSM з кроком out_s: колір у точці p береться з мозаїки в точці A·p − залишок(p)."""
    out = np.zeros((H, W, 3), np.uint8)
    for y0 in range(0, H, chunk):
        hh = min(chunk, H - y0)
        yy, xx = np.mgrid[y0:y0 + hh, 0:W].astype(np.float32)
        X = ox0 + xx * out_s; Y = oy0 + yy * out_s
        # грубий залишок на рідкій сітці
        cs = 16
        Xs, Ys = X[::cs, ::cs], Y[::cs, ::cs]
        r = rbf(np.stack([Xs.ravel(), Ys.ravel()], 1)).reshape(Xs.shape + (2,))
        r = np.stack([cv2.resize(r[..., k].astype(np.float32), (W, hh), interpolation=cv2.INTER_LINEAR) for k in range(2)], -1)
        qx = A[0, 0] * X + A[0, 1] * Y + A[0, 2] - r[..., 0]
        qy = A[1, 0] * X + A[1, 1] * Y + A[1, 2] - r[..., 1]
        mx = ((qx - g.x0) / g.s).astype(np.float32); my = ((qy - g.y0) / g.s).astype(np.float32)
        out[y0:y0 + hh] = cv2.remap(mos, mx, my, cv2.INTER_LINEAR, borderValue=0)
    return out


if __name__ == '__main__':
    tiles = sorted(glob.glob('ref/sat_tiles/tile*.jpg'))
    A, rbf, P, S, inl, g = main('data/osm_v2', 'ref/satellite_wide.jpg', tiles, 'build')
    mos = np.load('build/sat_mosaic_init.npy')
    al = aligned(mos, g, A, rbf, 0.5, g.x0, g.y0, g.W, g.H)
    np.save('build/sat_aligned_05.npy', al)
    print('done')
