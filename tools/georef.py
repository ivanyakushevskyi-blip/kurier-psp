"""Прив'язка супутникових знімків до сітки OSM.
1) широкий знімок (ref/satellite_wide.jpg) -> координати OSM: підбір афінного перетворення за збігом контурів
   будинків/доріг з OSM і перепадів яскравості на знімку;
2) кожен детальний тайл -> широкий знімок (SIFT), далі -> OSM;
Результат: build/georef.json (матриці «піксель тайла -> метри OSM»)."""
import glob, json, math, sys, xml.etree.ElementTree as ET
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

LAT0, LON0 = 49.422, 26.995
MLAT = 111320.0; MLON = MLAT * math.cos(math.radians(LAT0))
GX0, GY0, GW, GH, GS = -1500.0, -1700.0, 3600.0, 3000.0, 2.0     # сітка для узгодження, метри (x схід, y південь)


def m2g(x, y): return ((x - GX0) / GS, (y - GY0) / GS)


def osm_edges(osm_dir):
    nodes, ways = {}, []
    for f in sorted(glob.glob(osm_dir + '/t*.osm')):
        r = ET.parse(f).getroot()
        for n in r.iter('node'): nodes[n.get('id')] = (float(n.get('lat')), float(n.get('lon')))
        for w in r.iter('way'): ways.append(([nd.get('ref') for nd in w.iter('nd')], {x.get('k'): x.get('v') for x in w.iter('tag')}))
    W, H = int(GW / GS), int(GH / GS)
    im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
    RW = {'primary': 14, 'secondary': 12, 'tertiary': 10, 'residential': 7, 'unclassified': 7, 'trunk': 14}
    for nds, t in ways:
        p = [m2g((nodes[n][1] - LON0) * MLON, -(nodes[n][0] - LAT0) * MLAT) for n in nds if n in nodes]
        if len(p) < 2: continue
        if 'building' in t and len(p) > 2: d.polygon(p, outline=255)
        h = t.get('highway')
        if h in RW:
            d.line(p, fill=128, width=max(1, int(RW[h] / GS)))
    a = np.array(im).astype(np.float32)
    road = (a == 128)
    e = (a == 255).astype(np.float32) + (road & ~ndi.binary_erosion(road)).astype(np.float32)
    return ndi.gaussian_filter(e, 1.0)


def sat_edges(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy)
    return np.minimum(m / (np.percentile(m, 95) + 1e-6), 1.5)


def warp_to_grid(img_feat, A_px2m):
    """A_px2m: 2x3 піксель->метри. Повертає зображення на сітці."""
    S = np.array([[1 / GS, 0, -GX0 / GS], [0, 1 / GS, -GY0 / GS], [0, 0, 1]])
    M = (S @ np.vstack([A_px2m, [0, 0, 1]]))[:2]
    return cv2.warpAffine(img_feat, M, (int(GW / GS), int(GH / GS)), flags=cv2.INTER_LINEAR, borderValue=0), M


def ncc(a, b, mask):
    a = a[mask]; b = b[mask]
    a = (a - a.mean()) / (a.std() + 1e-6); b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def fit_wide(osm_dir, wide_path, A0):
    E = osm_edges(osm_dir)
    img = cv2.imread(wide_path)
    F = sat_edges(img)
    best = None
    for rot in np.arange(-2.0, 2.01, 0.25):
        for sc in np.arange(0.96, 1.041, 0.01):
            c, s = math.cos(math.radians(rot)) * sc, math.sin(math.radians(rot)) * sc
            L = np.array([[c, -s], [s, c]]) @ A0[:, :2]
            cx, cy = img.shape[1] / 2, img.shape[0] / 2
            t = A0 @ [cx, cy, 1] - L @ [cx, cy]
            A = np.hstack([L, t[:, None]])
            Wf, _ = warp_to_grid(F, A)
            mask = Wf > 0
            # зсув — фазовою кореляцією
            (dx, dy), resp = cv2.phaseCorrelate((E * mask).astype(np.float32), Wf.astype(np.float32))
            A2 = A.copy(); A2[:, 2] -= np.array([dx, dy]) * GS
            Wf2, _ = warp_to_grid(F, A2)
            sc_ = ncc(Wf2, E, Wf2 > 0)
            if best is None or sc_ > best[0]: best = (sc_, rot, sc, A2)
    print('wide fit: ncc %.3f rot %.2f scale %.3f' % best[:3])
    # дрібне уточнення зсуву ±6 м
    A = best[3]; b0 = best[0]
    for it in range(2):
        for dx in np.arange(-6, 6.1, 1.0):
            for dy in np.arange(-6, 6.1, 1.0):
                A2 = A.copy(); A2[:, 2] += [dx, dy]
                Wf2, _ = warp_to_grid(F, A2); v = ncc(Wf2, E, Wf2 > 0)
                if v > b0: b0, Abest = v, A2
        A = Abest if 'Abest' in dir() else A
    print('refined ncc %.3f' % b0)
    return A


def fit_tiles(wide_path, A_wide, tiles, norm_w=2576):
    wide = cv2.imread(wide_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=30000)
    kw, dw = sift.detectAndCompute(wide, None)
    fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    out = {}
    Aw = np.vstack([A_wide, [0, 0, 1]])
    for f in tiles:
        t0 = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        s = norm_w / t0.shape[1]
        t = cv2.resize(t0, None, fx=s * 0.5, fy=s * 0.5, interpolation=cv2.INTER_AREA)
        kt, dt = sift.detectAndCompute(t, None)
        g = [a for a, b in fl.knnMatch(dt, dw, k=2) if a.distance < 0.75 * b.distance]
        src = np.float32([kt[a.queryIdx].pt for a in g]) / (s * 0.5)        # у пікселях оригіналу
        dst = np.float32([kw[a.trainIdx].pt for a in g])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        M = Aw @ H                                                          # оригінальний піксель -> метри
        out[f] = dict(H=M.tolist(), w=t0.shape[1], h=t0.shape[0], inliers=int(inl.sum()))
        print(f, 'inliers', int(inl.sum()))
    return out


if __name__ == '__main__':
    osm_dir, wide, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    tiles = sys.argv[4:]
    # початкове наближення з першого прототипу (знімок satellite.png ≈ межі старої карти)
    A0 = np.array([[1.656, 0, -850 - 880 * 1.656], [0, 1.656, -750 - 560 * 1.656]])
    A = fit_wide(osm_dir, wide, A0)
    res = dict(wide=dict(A=A.tolist()), tiles=fit_tiles(wide, A, tiles))
    json.dump(res, open(outp, 'w'), indent=1)
