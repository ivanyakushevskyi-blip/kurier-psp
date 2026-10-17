"""Малювання міста кодом (вид строго згори, 0,5 м/пікс).
Геометрія — OSM; вигляд — зі знімків (колір даху кожного будинку, дерева, двори).
Усе намальоване одним «пензлем»: однакові текстури, однакові правила для дахів, без склейок.
Зразок: python3 -I tools/citydraw.py sample <x_0.5> <y_0.5> <w> <h> <out.png>
"""
import sys, os, math, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

MPP = 0.5
SRC = 0.32
K = SRC / MPP

# ---------------------------------------------------------------- палітра (RGB)
OUT = (34, 34, 42)
ASPH = [(64, 68, 76), (74, 78, 86), (82, 86, 94)]
MARK = (226, 226, 214)
SIDE = [(150, 148, 144), (166, 164, 158), (178, 176, 170)]
CURB = (196, 194, 188)
PAVED = [(132, 132, 130), (146, 146, 142), (158, 158, 154)]
GRASS = [(74, 122, 50), (92, 142, 58), (110, 160, 68), (132, 176, 80)]
DRY = [(128, 132, 70), (148, 150, 84)]
DIRT = [(128, 100, 72), (146, 118, 84), (162, 134, 98)]
WATER = [(48, 92, 140), (62, 110, 160), (96, 146, 190)]
BALLAST = [(96, 88, 82), (112, 104, 96), (128, 120, 110)]
SLEEPER = (82, 62, 48); RAIL = (186, 188, 194)
PITCH = [(78, 140, 56), (92, 156, 64)]
TRACK = [(166, 76, 58), (184, 92, 70)]
LINE = (236, 236, 230)
TREE = {'g': [(30, 74, 40), (52, 104, 48), (78, 134, 58), (112, 164, 74)],
        'y': [(96, 82, 26), (156, 128, 34), (204, 166, 48), (232, 202, 96)],
        'o': [(104, 50, 24), (164, 82, 34), (210, 116, 48), (236, 156, 84)]}
ROOFS = {   # базовий колір; відтінки рахуються
    'red': (176, 72, 50), 'darkred': (132, 56, 44), 'orange': (204, 108, 58), 'brown': (116, 84, 64),
    'gray': (136, 140, 146), 'darkgray': (92, 96, 104), 'metal': (182, 188, 194), 'white': (214, 216, 216),
    'green': (74, 124, 96), 'blue': (66, 100, 166), 'teal': (70, 118, 128), 'beige': (180, 162, 132)}


def shades(c):
    c = np.array(c, float)
    return [tuple(np.clip(c * 0.62, 0, 255).astype(int)), tuple(np.clip(c * 0.8, 0, 255).astype(int)),
            tuple(c.astype(int)), tuple(np.clip(c * 1.12 + 14, 0, 255).astype(int)), tuple(np.clip(c * 1.25 + 30, 0, 255).astype(int))]


def roof_kind(rgb):
    if rgb is None: return 'gray'
    c = np.array(rgb, float)
    hsv = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2HSV)[0, 0].astype(float)
    hsv[1] = min(255, hsv[1] * 1.4); hsv[2] = min(255, hsv[2] * 1.15 + 10)
    c = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0].astype(float)
    best, bd = None, 1e9
    for k, v in ROOFS.items():
        v = np.array(v, float)
        d = ((c - v) ** 2).sum() + 2.0 * (((c - c.mean()) - (v - v.mean())) ** 2).sum()
        if d < bd: best, bd = k, d
    return best


# ---------------------------------------------------------------- дані
_CACHE = {}


def data():
    if 'w' not in _CACHE:
        import citygen as CG
        nodes, ntags, ways = CG.load('data/osm_v2')
        _CACHE.update(w=ways, n=ntags, nodes=nodes, B=json.load(open('build/buildings.json'))['buildings'])
    return _CACHE


ROAD_W = {'primary': 13, 'secondary': 11, 'tertiary': 9.5, 'residential': 7, 'unclassified': 6.5, 'living_street': 5.5,
          'primary_link': 7, 'secondary_link': 6.5, 'tertiary_link': 6.5, 'service': 4.5, 'track': 3.5}
SIDE_W = {'primary': 3.0, 'secondary': 3.0, 'tertiary': 2.6, 'residential': 2.0, 'unclassified': 1.8, 'living_street': 1.6,
          'primary_link': 2.0, 'secondary_link': 2.0, 'tertiary_link': 2.0}


def dither(H, W, seed=1):
    rs = np.random.RandomState(seed)
    n = cv2.GaussianBlur(rs.rand(H // 4 + 2, W // 4 + 2).astype(np.float32), (0, 0), 1.0)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_NEAREST)[:H, :W]
    n = (n - n.min()) / (n.max() - n.min() + 1e-6)
    fine = rs.rand(H, W).astype(np.float32)
    return n, fine


# ---------------------------------------------------------------- рендер вікна
def render(x0, y0, W, H):
    D = data()
    to = lambda p: (p[0] * K - x0, p[1] * K - y0)

    def visible(pts, m=60):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return not (max(xs) < -m or min(xs) > W + m or max(ys) < -m or min(ys) > H + m)

    ways = []
    for w in D['w']:
        pts = [to(p) for p in w['pts']]
        if visible(pts): ways.append((w, pts))

    # ---------- класи землі
    G_YARD, G_GRASS, G_WATER, G_SIDE, G_ROAD, G_PATH, G_PED, G_RAIL, G_DIRT, G_PITCH, G_TRACK, G_PARK, G_PAVED, G_WOOD = range(1, 15)
    im = Image.new('L', (W, H), G_YARD); dr = ImageDraw.Draw(im)
    areas = []
    for w, pts in ways:
        if not w['closed']: continue
        t = w['t']; c = None
        lu, le, na, am = t.get('landuse'), t.get('leisure'), t.get('natural'), t.get('amenity')
        if lu in ('retail', 'commercial', 'industrial', 'garages', 'railway'): c = G_PAVED
        elif lu in ('construction', 'brownfield'): c = G_DIRT
        elif lu in ('grass', 'meadow', 'recreation_ground', 'village_green', 'flowerbed', 'cemetery') or le in ('park', 'garden', 'playground', 'dog_park'): c = G_GRASS
        elif le == 'track': c = G_TRACK
        elif le == 'pitch': c = G_PITCH
        elif le in ('stadium', 'sports_centre'): c = G_PAVED
        elif na in ('wood', 'scrub') or lu == 'forest': c = G_WOOD
        elif na in ('grassland', 'heath'): c = G_GRASS
        elif na == 'water' or lu in ('reservoir', 'basin') or le == 'swimming_pool': c = G_WATER
        elif am == 'parking' and t.get('parking') not in ('underground', 'multi-storey'): c = G_PARK
        elif (t.get('highway') == 'pedestrian' and t.get('area') == 'yes') or t.get('place') == 'square' or t.get('area:highway') == 'pedestrian': c = G_PED
        elif t.get('railway') == 'platform': c = G_PAVED
        if c is not None:
            a = 0.5 * abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts))))
            areas.append((a, c, pts))
    areas.sort(key=lambda a: -a[0])
    for _, c, pts in areas: dr.polygon(pts, fill=c)
    # знімок: двори (трава/плитка/ґрунт) там, де OSM нічого не каже
    sat = np.load('build/sat_map.npy', mmap_mode='r')
    sx0, sy0 = int(x0 / K), int(y0 / K); sx1, sy1 = int((x0 + W) / K) + 2, int((y0 + H) / K) + 2
    raw = np.ascontiguousarray(sat[max(0, sy0):sy1, max(0, sx0):sx1])
    sw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32)   # RGB, ~0.5 м
    gray32 = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.resize(np.abs(cv2.Laplacian(cv2.GaussianBlur(gray32, (3, 3), 0), cv2.CV_32F)), (W, H), interpolation=cv2.INTER_AREA)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.5)
    sm = cv2.medianBlur(sw.astype(np.uint8), 5).astype(np.float32)
    r, g, bl = sm[..., 0], sm[..., 1], sm[..., 2]
    mx = sm.max(-1); ch = mx - sm.min(-1); exg = 2 * g - r - bl
    dark = mx < 62
    ycls = np.full((H, W), 3, np.uint8)                          # 1 трава, 2 ґрунт, 3 плитка/асфальт
    ycls[exg > 6] = 1
    ycls[(r > bl + 28) & (exg <= 6) & (ch > 34) & (mx > 110)] = 2
    ycls[dark] = 0
    if (ycls == 0).any():
        _, (iy, ix) = ndi.distance_transform_edt(ycls == 0, return_indices=True); ycls = ycls[iy, ix]
    oh = np.stack([cv2.blur((ycls == k).astype(np.float32), (9, 9)) for k in (1, 2, 3)])
    ycls = oh.argmax(0).astype(np.uint8) + 1
    cls = np.array(im)
    yard = cls == G_YARD
    cls[yard & (ycls == 1)] = G_GRASS; cls[yard & (ycls == 2)] = G_DIRT; cls[yard & (ycls == 3)] = G_PAVED

    # лінійні: вода, рейки, тротуари, доріжки, дороги
    im = Image.fromarray(cls); dr = ImageDraw.Draw(im)
    def line(pts, wm, fill, d=dr):
        wp = max(1.0, wm / MPP); d.line(pts, fill=fill, width=int(round(wp)))
        rr = wp / 2
        for x, y in pts: d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill)
    WW = {'river': 20, 'canal': 10, 'stream': 4, 'ditch': 2, 'drain': 2}
    rails, drive, foot = [], [], []
    for w, pts in ways:
        t = w['t']
        if t.get('waterway') in WW and t.get('tunnel') not in ('yes', 'culvert'): line(pts, WW[t['waterway']], G_WATER)
        if t.get('railway') in ('rail', 'siding', 'narrow_gauge') and t.get('tunnel') != 'yes':
            line(pts, 4.0, G_RAIL); rails.append(pts)
        h = t.get('highway')
        if h and t.get('tunnel') != 'yes' and t.get('area') != 'yes':
            if h in ROAD_W: drive.append((h, pts, w))
            elif h in ('footway', 'path', 'cycleway', 'steps', 'pedestrian', 'bridleway'): foot.append((h, pts, w))
    for h, pts, w in drive:
        if h in SIDE_W and w['t'].get('sidewalk') != 'no': line(pts, ROAD_W[h] + 2 * SIDE_W[h], G_SIDE)
    for h, pts, w in foot:
        line(pts, 8 if h == 'pedestrian' else 2.0, G_PED if h == 'pedestrian' else G_PATH)
    for h, pts, w in drive: line(pts, ROAD_W[h], G_ROAD)
    cls = np.array(im)

    # ---------- текстури землі
    img = np.zeros((H, W, 3), np.uint8)
    blob, fine = dither(H, W, 7)
    yy, xx = np.mgrid[0:H, 0:W]
    def pick(ramp, t):
        idx = np.clip((t * len(ramp)).astype(int), 0, len(ramp) - 1); return np.array(ramp, np.uint8)[idx]
    def put(cid, arr):
        m = cls == cid; img[m] = arr[m]
    t_grass = 0.25 + 0.5 * blob + 0.25 * (fine > 0.86) - 0.15 * (fine < 0.08)
    put(G_GRASS, pick(GRASS[1:], t_grass)); put(G_WOOD, pick(GRASS[:3], t_grass))
    dry = (cls == G_GRASS) & (sm[..., 0] > sm[..., 1] + 4)
    img[dry] = pick(DRY, blob)[dry]
    put(G_DIRT, pick(DIRT, 0.3 + 0.5 * blob + 0.2 * (fine > 0.9)))
    tile6 = ((xx % 6 == 0) | (yy % 6 == 0))
    put(G_PAVED, pick(PAVED, 0.35 + 0.3 * blob + 0.2 * (fine > 0.92)))
    tile4 = ((xx % 4 == 0) | (yy % 4 == 0))
    put(G_SIDE, np.where(tile4[..., None], np.array(SIDE[1], np.uint8), np.array(SIDE[2], np.uint8)))
    brick = (yy % 3 == 0) | (((xx + (yy // 3) % 2 * 3) % 6) == 0)
    put(G_PED, np.where(brick[..., None], np.array(SIDE[0], np.uint8), np.array(SIDE[2], np.uint8)))
    put(G_PATH, np.full((H, W, 3), SIDE[1], np.uint8))
    asph = pick(ASPH, 0.35 + 0.35 * blob + 0.3 * (fine > 0.93))
    put(G_ROAD, asph); put(G_PARK, pick(ASPH[1:], 0.4 + 0.4 * blob))
    rip = (np.sin(xx * 0.5 + yy * 0.15 + blob * 4) > 0.85)
    put(G_WATER, np.where(rip[..., None], np.array(WATER[2], np.uint8), pick(WATER[:2], 0.3 + 0.5 * blob)))
    put(G_RAIL, pick(BALLAST, fine))
    put(G_PITCH, np.where(((xx // 8) % 2 == 0)[..., None], np.array(PITCH[0], np.uint8), np.array(PITCH[1], np.uint8)))
    put(G_TRACK, np.full((H, W, 3), TRACK[1], np.uint8))
    put(G_YARD, pick(PAVED, 0.5 + 0.0 * blob))
    road = cls == G_ROAD
    walk = np.isin(cls, (G_SIDE, G_PED))
    img[walk & cv2.dilate(road.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = CURB
    soft = np.isin(cls, (G_GRASS, G_WOOD, G_DIRT))
    img[walk & cv2.dilate(soft.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = SIDE[0]
    # рейки
    ov = Image.new('L', (W, H), 0); do = ImageDraw.Draw(ov)
    for pts in rails:
        for i in range(len(pts) - 1):
            (xa, ya), (xb, yb) = pts[i], pts[i + 1]; L = math.hypot(xb - xa, yb - ya)
            if L < 1: continue
            ux, uy = (xb - xa) / L, (yb - ya) / L; s = 0.0
            while s < L:
                cx, cy = xa + ux * s, ya + uy * s
                do.line([(cx + uy * 3.5, cy - ux * 3.5), (cx - uy * 3.5, cy + ux * 3.5)], fill=1, width=1); s += 2.5
        for off in (-1.6, 1.6):
            sh = []
            for i in range(len(pts)):
                a = pts[max(i - 1, 0)]; b2 = pts[min(i + 1, len(pts) - 1)]
                L = math.hypot(b2[0] - a[0], b2[1] - a[1]) or 1
                sh.append((pts[i][0] - (b2[1] - a[1]) / L * off, pts[i][1] + (b2[0] - a[0]) / L * off))
            do.line(sh, fill=2, width=1)
    ova = np.array(ov); img[ova == 1] = SLEEPER; img[ova == 2] = RAIL
    # розмітка, переходи
    mk = Image.new('L', (W, H), 0); dm = ImageDraw.Draw(mk)
    for h, pts, w in drive:
        if h not in ('primary', 'secondary', 'tertiary') or w['t'].get('oneway') == 'yes': continue
        s0 = 0.0
        for i in range(len(pts) - 1):
            (xa, ya), (xb, yb) = pts[i], pts[i + 1]; L = math.hypot(xb - xa, yb - ya); s = 0.0
            while s < L:
                ph = (s0 + s) % 10
                if ph < 5:
                    e = min(s + 5 - ph, L)
                    dm.line([(xa + (xb - xa) * s / L, ya + (yb - ya) * s / L), (xa + (xb - xa) * e / L, ya + (yb - ya) * e / L)], fill=1)
                    s = e
                else: s += 10 - ph
            s0 += L
    node_road = {}
    for h, pts, w in drive:
        for k, nid in enumerate(w['nds']): node_road.setdefault(nid, (h, pts, k))
    for nid, t in D['n'].items():
        if t.get('highway') != 'crossing' or nid not in node_road: continue
        h, pts, k = node_road[nid]
        a = pts[max(k - 1, 0)]; b2 = pts[min(k + 1, len(pts) - 1)]
        L = math.hypot(b2[0] - a[0], b2[1] - a[1]) or 1; ux, uy = (b2[0] - a[0]) / L, (b2[1] - a[1]) / L
        cx, cy = pts[k]; hw = ROAD_W[h] / MPP / 2; v = -hw + 1
        while v < hw - 0.5:
            dm.line([(cx - uy * v - ux * 3, cy + ux * v - uy * 3), (cx - uy * v + ux * 3, cy + ux * v + uy * 3)], fill=1, width=2); v += 3
    img[(np.array(mk) > 0) & road] = MARK
    # бігова доріжка: поле всередині стадіону -> «овал» навколо поля
    stad = [np.array(p, np.float32) for w, p in ways if w['closed'] and w['t'].get('leisure') == 'stadium']
    for w, pts in ways:
        if not (w['closed'] and w['t'].get('leisure') == 'pitch' and len(pts) >= 4): continue
        P = np.array(pts, np.float32); c = P.mean(0)
        if not any(cv2.pointPolygonTest(S_, (float(c[0]), float(c[1])), False) >= 0 for S_ in stad): continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(P)
        L, Sx = max(rw, rh), min(rw, rh)
        a = math.radians(ang) if rw >= rh else math.radians(ang + 90)
        ux, uy = math.cos(a), math.sin(a); vx, vy = -uy, ux
        m = np.zeros((H, W), np.uint8)
        half = L / 2 - Sx * 0.22
        R_in = math.hypot(L / 2 - half, Sx / 2) + 2
        for k, off in enumerate((9.0 / MPP, 0.0)):
            R = R_in + off
            poly = []
            for t in np.linspace(-90, 90, 24):
                tr = math.radians(t); poly.append((cx + ux * (half + R * math.cos(tr)) + vx * R * math.sin(tr), cy + uy * (half + R * math.cos(tr)) + vy * R * math.sin(tr)))
            for t in np.linspace(90, 270, 24):
                tr = math.radians(t); poly.append((cx + ux * (-half + R * math.cos(tr)) + vx * R * math.sin(tr), cy + uy * (-half + R * math.cos(tr)) + vy * R * math.sin(tr)))
            cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 1 if k == 0 else 2)
        trk = m == 1
        img[trk] = TRACK[1]
        dd = cv2.distanceTransform((m == 1).astype(np.uint8), cv2.DIST_L2, 3)
        img[trk & (np.round(dd) % 3 == 0) & (dd > 1)] = (214, 170, 150)
        img[m == 2] = np.where(((xx // 8) % 2 == 0)[..., None], np.array(PITCH[0], np.uint8), np.array(PITCH[1], np.uint8))[m == 2]
        cls[m > 0] = G_PITCH
    # спортивні поля: білі лінії по контуру + середня лінія
    for w, pts in ways:
        if w['closed'] and w['t'].get('leisure') == 'pitch' and len(pts) >= 4:
            P = np.array(pts, np.float32); rect = cv2.minAreaRect(P); bx = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(img, [bx], True, LINE, 1)
            (cx, cy), (rw, rh), ang = rect; a = math.radians(ang)
            if rw >= rh: ux, uy = -math.sin(a), math.cos(a); half = rh / 2
            else: ux, uy = math.cos(a), math.sin(a); half = rw / 2
            cv2.line(img, (int(cx - ux * half), int(cy - uy * half)), (int(cx + ux * half), int(cy + uy * half)), LINE, 1)
            cv2.circle(img, (int(cx), int(cy)), max(3, int(min(rw, rh) * 0.12)), LINE, 1)
        if w['closed'] and w['t'].get('leisure') == 'track' and len(pts) >= 4:
            P = np.array(pts, np.int32)
            m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [P], 1)
            d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
            lane = (m > 0) & (np.round(d) % 3 == 0) & (d > 1)
            img[lane] = TRACK[0]

    # ---------- будинки
    blds = []
    for i, b in enumerate(D['B']):
        pts = [to(p) for p in b['q']]
        if not visible(pts, 30): continue
        blds.append((i, b, pts))
    bmask = np.zeros((H, W), np.uint8)
    for i, b, pts in blds: cv2.fillPoly(bmask, [np.round(np.array(pts)).astype(np.int32)], 1)
    bmask = bmask.astype(bool)

    # ---------- дерева зі знімка
    green = exg > 8; autumn = (r > bl + 30) & (ch > 34)
    canopy = (tex > 4.0) & (green | autumn | ((ch > 18) & (mx < 120) & (tex > 7))) & ~(mx < 40)
    park = np.zeros((H, W), np.uint8)
    for w, pts in ways:
        if w['closed'] and (w['t'].get('leisure') == 'park' or w['t'].get('natural') == 'wood' or w['t'].get('landuse') == 'forest'):
            cv2.fillPoly(park, [np.round(np.array(pts)).astype(np.int32)], 1)
    park = park.astype(bool) & ~np.isin(cls, (G_PATH, G_PED, G_ROAD, G_SIDE))
    canopy |= park & ((tex > 3.0) | (mx < 70)) & (exg > -8)
    canopy &= ~cv2.dilate(bmask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    canopy &= ~np.isin(cls, (G_ROAD, G_RAIL, G_WATER, G_PITCH, G_TRACK))
    canopy = cv2.morphologyEx(canopy.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    canopy = cv2.morphologyEx(canopy, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    dist = cv2.distanceTransform(canopy.astype(np.uint8), cv2.DIST_L2, 5)
    pk = (dist == ndi.maximum_filter(dist, size=9)) & (dist >= 3.0)
    ys, xs = np.nonzero(pk); order = np.argsort(-dist[ys, xs])
    trees = []
    for y, x in zip(ys[order], xs[order]):
        rr = float(min(dist[y, x] * 1.25 + 2, 13))
        if any((tx - x) ** 2 + (ty - y) ** 2 < (0.75 * (tr + rr)) ** 2 for tx, ty, tr, _ in trees): continue
        c = sm[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3).mean(0)
        kind = 'g' if c[1] > c[0] + 2 else ('o' if c[0] > c[1] + 24 else 'y')
        trees.append((int(x), int(y), rr, kind))

    # ---------- тіні (на південний схід)
    shadow = np.zeros((H, W), np.uint8)
    for i, b, pts in blds:
        s = int(np.clip(b['h'] * 0.22, 1, 7))
        P = np.round(np.array(pts)).astype(np.int32)
        for k in range(1, s + 1): cv2.fillPoly(shadow, [P + [k, k]], 1)
    for x, y, rr, _ in trees: cv2.circle(shadow, (x + 2, y + 2), int(rr), 1, -1)
    sh = shadow.astype(bool) & ~bmask
    img[sh] = (img[sh] * 0.7).astype(np.uint8)

    # ---------- дахи
    rs_all = random.Random(7)
    for i, b, pts in sorted(blds, key=lambda z: z[1]['h']):
        draw_roof(img, b, pts, random.Random(i * 31 + 7))

    # ---------- дерева
    for k, (x, y, rr, kind) in enumerate(sorted(trees, key=lambda t: t[1])):
        draw_tree(img, x, y, rr, kind, random.Random(k * 13 + x))
    return img, cls


def draw_roof(img, b, pts, rs):
    H, W = img.shape[:2]
    P = np.array(pts, np.float32)
    bx0, by0 = np.floor(P.min(0)).astype(int) - 3; bx1, by1 = np.ceil(P.max(0)).astype(int) + 4
    if bx1 < 0 or by1 < 0 or bx0 >= W or by0 >= H: return
    w, h = bx1 - bx0, by1 - by0
    m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(P - [bx0, by0]).astype(np.int32)], 1)
    F = m.astype(bool)
    if F.sum() < 4: return
    area = b['area']; t = b['t']; bt = t.get('building', 'yes')
    per = float(np.sum(np.linalg.norm(np.diff(np.array(b['q']), axis=0), axis=1))) * SRC
    circ = 4 * math.pi * area / max(per, 1) ** 2
    kind = roof_kind(b.get('roof'))
    S = [np.array(c, np.uint8) for c in shades(ROOFS[kind])]
    out = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    if circ > 0.9 and area > 250:                       # купол / кругла будівля
        cx, cy = (P - [bx0, by0]).mean(0)
        ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360
        rib = (ang // 7) % 2 == 0
        lit = np.cos(np.radians(ang - 225))
        base = np.where(lit[..., None] > 0.3, S[3], np.where(lit[..., None] < -0.3, S[1], S[2]))
        out[:] = np.where(rib[..., None], base, np.clip(base.astype(int) + 18, 0, 255).astype(np.uint8))
        R = d.max()
        out[d > R * 0.82] = S[4]; out[d > R * 0.9] = S[0]
        out[(d > R * 0.55) & (d < R * 0.62)] = S[1]
    elif (bt in ('house', 'detached', 'semidetached_house', 'terrace', 'church') or area < 260 or (area < 700 and b['h'] < 11)) \
            and bt not in ('garage', 'garages', 'shed', 'kiosk'):
        # скатний (вальмовий) дах: світло з північного заходу
        dd = cv2.GaussianBlur(d, (0, 0), 0.8)
        gy, gx = np.gradient(dd)
        nrm = np.hypot(gx, gy) + 1e-6
        lit = (-gx - gy) / nrm / 1.414
        ridge = (nrm < 0.25) & (d > 1.5)
        out[:] = S[2]
        out[lit > 0.35] = S[3]; out[lit < -0.35] = S[1]
        out[ridge] = S[4]
    elif bt in ('garage', 'garages', 'shed', 'industrial', 'warehouse', 'roof', 'service') or kind == 'metal':
        # профнастил: смуги вздовж довшої сторони
        rect = cv2.minAreaRect(P); (rw, rh), a = rect[1], math.radians(rect[2])
        ux, uy = (math.cos(a), math.sin(a)) if rw < rh else (-math.sin(a), math.cos(a))
        st = ((xx * ux + yy * uy) // 2) % 2 == 0
        out[:] = np.where(st[..., None], S[2], S[3])
        out[(d >= 1) & (d < 2)] = S[1]
    else:
        # плаский дах: бортик, обладнання
        out[:] = S[2]
        out[(d >= 1) & (d < 2.2)] = S[3]
        out[(d >= 2.2) & (d < 3.2)] = S[1]
        ys, xs = np.nonzero(d > 6)
        for _ in range(min(len(xs) // 400, 5)):
            k = rs.randrange(len(xs)); vx, vy = xs[k], ys[k]
            out[vy:vy + 3, vx:vx + 4] = S[3]; out[vy + 3, vx + 1:vx + 5] = S[1]
    out[F & (d < 1.01)] = S[0] if circ > 0.9 else out[F & (d < 1.01)]
    ol = cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool) & ~F
    dst = img[max(0, by0):min(H, by1), max(0, bx0):min(W, bx1)]
    oy, ox = max(0, by0) - by0, max(0, bx0) - bx0
    Fs = F[oy:oy + dst.shape[0], ox:ox + dst.shape[1]]; Os = ol[oy:oy + dst.shape[0], ox:ox + dst.shape[1]]
    dst[Fs] = out[oy:oy + dst.shape[0], ox:ox + dst.shape[1]][Fs]
    dst[Os] = OUT


def draw_tree(img, x, y, rr, kind, rs):
    H, W = img.shape[:2]
    R = int(math.ceil(rr)) + 2
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    m = xx ** 2 + yy ** 2 <= (rr * 0.82) ** 2
    for _ in range(6):
        a = rs.random() * 6.283; dd = rr * (0.3 + rs.random() * 0.3); lr = rr * (0.35 + rs.random() * 0.2)
        m |= (xx - math.cos(a) * dd) ** 2 + (yy - math.sin(a) * dd) ** 2 <= lr ** 2
    C = [np.array(c, np.uint8) for c in TREE[kind]]
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    lit = (-xx - yy) / (rr + 1e-3)
    o = np.zeros(m.shape + (3,), np.uint8); o[:] = C[2]
    o[lit < -0.25] = C[1]; o[(lit > 0.35) & (dist > 1.5)] = C[3]
    spk = (np.array([[rs.random() for _ in range(m.shape[1])] for _ in range(m.shape[0])]) < 0.06) & (lit > 0)
    o[spk] = C[3]
    edge = m & (dist < 1.01); o[edge] = C[0]
    ol = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
    x0, y0 = x - R, y - R
    for yy_ in range(m.shape[0]):
        Y = y0 + yy_
        if not 0 <= Y < H: continue
        for xx_ in np.nonzero(m[yy_] | ol[yy_])[0]:
            X = x0 + xx_
            if 0 <= X < W: img[Y, X] = o[yy_, xx_] if m[yy_, xx_] else OUT


if __name__ == '__main__':
    if sys.argv[1] == 'sample':
        cx, cy, w, h = map(int, sys.argv[2:6])
        img, _ = render(cx - w // 2, cy - h // 2, w, h)
        cv2.imwrite(sys.argv[6], img[..., ::-1])
