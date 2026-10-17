"""Збирання карти з цілих знімків від GPT у масштабі гри (0,5 м/пікс).
- кожен квартал (простір між дорогами OSM) береться цілком з одного знімка;
- дороги, тротуари, розмітка, переходи малюються кодом за OSM поверх (шви ховаються під дорогами);
- усе зводиться до палітри ref/gen/palette_v3.txt (+ кольори доріг).
python3 -I tools/gencompose.py <out.png> name1 name2 ...   (ref/gen/<name>.png + build/gen/<name>.npz)
"""
import sys, os, glob, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

MPP_SRC = 0.32           # сітка, до якої прив'язані картинки (build/sat_map)
MPP = 0.5                # масштаб гри
K = MPP_SRC / MPP

ROAD_W = {'primary': 14, 'secondary': 12, 'tertiary': 10, 'residential': 7.5, 'unclassified': 7, 'living_street': 6,
          'primary_link': 8, 'secondary_link': 7, 'tertiary_link': 7, 'service': 4.5}
SIDE_W = {'primary': 3.0, 'secondary': 3.0, 'tertiary': 2.6, 'residential': 2.2, 'unclassified': 2.0, 'living_street': 1.6,
          'primary_link': 2.0, 'secondary_link': 2.0, 'tertiary_link': 2.0}
CUT = ('primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'living_street', 'primary_link', 'secondary_link', 'tertiary_link')

C_ASPH = (0x55, 0x59, 0x5F); C_PAVE = (0xA3, 0xA5, 0xA8); C_CURB = (0xC4, 0xC6, 0xC8); C_MARK = (0xE6, 0xE6, 0xDC)


def load_palette():
    cols = []
    for line in open('ref/gen/palette_v3.txt'):
        hx = line.split()[0]
        cols.append(tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5)))
    for c in (C_MARK,):
        if c not in cols: cols.append(c)
    return np.array(cols, np.float32)          # RGB


def load(name):
    p = [f for f in glob.glob('ref/gen/%s.*' % name) if f.endswith(('.png', '.jpg'))][0]
    return cv2.imread(p), np.load('build/gen/%s.npz' % name)['H']


def osm_ways():
    import citygen as CG
    _, ntags, ways = CG.load('data/osm_v2')
    return ntags, ways


def compose(names, crop=None):
    items = [load(n) for n in names]
    S = np.diag([K, K, 1.0])                                       # 0.32-сітка -> 0.5-сітка
    cs = []
    for img, H in items:
        h, w = img.shape[:2]
        cs.append(cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), S @ H)[0])
    allc = np.vstack(cs)
    x0, y0 = np.floor(allc.min(0)).astype(int); x1, y1 = np.ceil(allc.max(0)).astype(int)
    if crop: x0, y0, x1, y1 = crop
    W, Hh = x1 - x0, y1 - y0
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    warped = []
    for img, H in items:
        w = cv2.warpPerspective(img, T @ S @ H, (W, Hh), flags=cv2.INTER_NEAREST)
        m = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), T @ S @ H, (W, Hh), flags=cv2.INTER_NEAREST) > 0
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        warped.append((w, m))
    # ---- дороги з OSM у координатах вікна
    ntags, ways = osm_ways()
    to = lambda p: ((p[0] * K - x0), (p[1] * K - y0))
    roads, cutl = [], []
    for wy in ways:
        hw = wy['t'].get('highway')
        if hw not in ROAD_W or wy['t'].get('tunnel') == 'yes' or wy['t'].get('area') == 'yes': continue
        pts = [to(p) for p in wy['pts']]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if max(xs) < -50 or min(xs) > W + 50 or max(ys) < -50 or min(ys) > Hh + 50: continue
        roads.append((hw, pts, wy))
    def raster(fn):
        im = Image.new('L', (W, Hh), 0); dr = ImageDraw.Draw(im); fn(dr); return np.array(im, bool)
    def line(dr, pts, wm):
        wp = max(1.0, wm / MPP)
        dr.line(pts, fill=1, width=int(round(wp)))
        r = wp / 2
        for x, y in pts: dr.ellipse([x - r, y - r, x + r, y + r], fill=1)
    side = raster(lambda dr: [line(dr, p, ROAD_W[h] + 2 * SIDE_W[h]) for h, p, _ in roads if h in SIDE_W])
    asph = raster(lambda dr: [line(dr, p, ROAD_W[h]) for h, p, _ in roads])
    cut = raster(lambda dr: [dr.line(p, fill=1, width=1) for h, p, _ in roads if h in CUT])
    # ---- квартали: області між осями доріг; кожен — з одного знімка
    nb, lab = cv2.connectedComponents((~cut).astype(np.uint8), connectivity=4)
    conf = [cv2.erode(m.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0 for _, m in warped]
    centers = [c.mean(0) - [x0, y0] for c in cs]
    ys, xs = np.nonzero(lab > 0); labs = lab[ys, xs]
    cnt = np.bincount(labs, minlength=nb).astype(float)
    cx = np.bincount(labs, xs, minlength=nb) / np.maximum(cnt, 1); cy = np.bincount(labs, ys, minlength=nb) / np.maximum(cnt, 1)
    best = np.full(nb, -1e9); choice = np.zeros(nb, int)
    for k, c in enumerate(conf):
        cov = np.bincount(labs, c[ys, xs], minlength=nb) / np.maximum(cnt, 1)
        sc = cov * 1e5 - np.hypot(cx - centers[k][0], cy - centers[k][1])
        u = sc > best; best[u] = sc[u]; choice[u] = k
    owner = choice[lab]
    _, (iy, ix) = ndi.distance_transform_edt(cut, return_indices=True)
    owner = owner[iy, ix]
    out = np.zeros((Hh, W, 3), np.uint8); have = np.zeros((Hh, W), bool)
    for k, (w, m) in enumerate(warped):
        sel = (owner == k) & m; out[sel] = w[sel]; have |= sel
    for k, (w, m) in enumerate(warped):
        sel = ~have & m; out[sel] = w[sel]; have |= sel
    # ---- палітра (квартали від GPT)
    pal = load_palette()
    rgb = out[..., ::-1].reshape(-1, 3).astype(np.float32)
    idx = np.concatenate([((rgb[i:i + 300000, None] - pal[None]) ** 2).sum(-1).argmin(1) for i in range(0, len(rgb), 300000)])
    out = pal[idx].reshape(out.shape).astype(np.uint8)[..., ::-1].copy()
    # ---- дороги поверх
    bgr = lambda c: c[::-1]
    out[side & ~asph] = bgr(C_PAVE)
    curb = side & ~asph & cv2.dilate(asph.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    out[curb] = bgr(C_CURB)
    out[asph] = bgr(C_ASPH)
    # розмітка (переривчаста) на великих вулицях, переходи
    mk = Image.new('L', (W, Hh), 0); dm = ImageDraw.Draw(mk)
    for h, pts, wy in roads:
        if h not in ('primary', 'secondary', 'tertiary') or wy['t'].get('oneway') == 'yes': continue
        s0 = 0.0
        for i in range(len(pts) - 1):
            (xa, ya), (xb, yb) = pts[i], pts[i + 1]; L = math.hypot(xb - xa, yb - ya); s = 0.0
            while s < L:
                ph = (s0 + s) % 12
                if ph < 6:
                    e = min(s + 6 - ph, L)
                    dm.line([(xa + (xb - xa) * s / L, ya + (yb - ya) * s / L), (xa + (xb - xa) * e / L, ya + (yb - ya) * e / L)], fill=1, width=1)
                    s = e
                else: s += 12 - ph
            s0 += L
    node_road = {}
    for h, pts, wy in roads:
        for k, nid in enumerate(wy['nds']): node_road.setdefault(nid, (h, pts, k))
    for nid, t in ntags.items():
        if t.get('highway') != 'crossing' or nid not in node_road: continue
        h, pts, k = node_road[nid]
        a = pts[max(k - 1, 0)]; b = pts[min(k + 1, len(pts) - 1)]
        L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1; ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        cx_, cy_ = pts[k]; hw = ROAD_W[h] / MPP / 2; v = -hw + 1
        while v < hw - 0.5:
            dm.line([(cx_ - uy * v - ux * 3, cy_ + ux * v - uy * 3), (cx_ - uy * v + ux * 3, cy_ + ux * v + uy * 3)], fill=1, width=2)
            v += 3.5
    mka = np.array(mk, bool) & asph
    out[mka] = bgr(C_MARK)
    out[~have & ~side & ~asph] = 0
    return out, (x0, y0), dict(asph=asph, side=side)


def car_sprite(length_px, width_px, ang_deg=0):
    """Тимчасовий спрайт Golf: сріблястий хетчбек, вигляд згори."""
    L, Wd = length_px, width_px
    im = np.zeros((Wd + 4, L + 4, 4), np.uint8)
    cv2.rectangle(im, (2, 2), (L + 1, Wd + 1), (60, 58, 56, 255), -1)
    cv2.rectangle(im, (3, 3), (L, Wd), (198, 200, 202, 255), -1)
    cv2.rectangle(im, (int(L * 0.55), 4), (int(L * 0.75), Wd - 1), (70, 62, 54, 255), -1)      # лобове скло
    cv2.rectangle(im, (int(L * 0.18), 4), (int(L * 0.28), Wd - 1), (70, 62, 54, 255), -1)      # заднє скло
    cv2.rectangle(im, (int(L * 0.3), 4), (int(L * 0.53), Wd - 1), (214, 216, 218, 255), -1)    # дах
    return im


if __name__ == '__main__':
    out, (x0, y0), _ = compose(sys.argv[2:])
    cv2.imwrite(sys.argv[1], out)
    print('saved', sys.argv[1], out.shape, 'origin (0.5 m grid)', x0, y0)
