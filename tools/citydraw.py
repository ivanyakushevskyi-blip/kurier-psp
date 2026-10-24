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
OSM_DIR = 'data/osm_v3'


def grid_pt(pt, grid='v1'):
    """Точка в сітці 0,32 м: ручні дані, задані в старій сітці (v1), зсуваються на цілий зсув нової сітки."""
    off = json.load(open('build/sat_map.json')).get('off', [0, 0]) if grid == 'v1' else [0, 0]
    return (float(pt[0] + off[0]), float(pt[1] + off[1]))


def data():
    if 'w' not in _CACHE:
        import citygen as CG
        nodes, ntags, ways = CG.load(OSM_DIR)
        # лише те, що в межах сітки (з запасом), + рамки для швидкого відбору у вікні
        meta = json.load(open('build/sat_map.json'))
        GW, GH = meta['W'], meta['H']
        keep = []
        for w in ways:
            a = np.array(w['pts'], np.float32)
            bb = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
            if bb[2] < -300 or bb[3] < -300 or bb[0] > GW + 300 or bb[1] > GH + 300: continue
            w['bb'] = bb; keep.append(w)
        ways = keep
        _CACHE.update(w=ways, n=ntags, nodes=nodes, B=json.load(open('build/buildings.json'))['buildings'],
                      SAT=(json.load(open('build/ms_buildings.json')) if os.path.exists('build/ms_buildings.json') else
                           json.load(open('build/sat_houses.json')) if os.path.exists('build/sat_houses.json') else []) +
                          (json.load(open('build/manual_buildings.json')) if os.path.exists('build/manual_buildings.json') else []))
        # ручні кольори дахів ключових будівель (data/roof_overrides.json, точки в сітці 0,32 м)
        if os.path.exists('data/roof_overrides.json'):
            for o in json.load(open('data/roof_overrides.json')):
                pt = grid_pt(o['pt'], o.get('grid', 'v1'))
                for b in _CACHE['B'] + _CACHE['SAT']:
                    if cv2.pointPolygonTest(np.array(b['q'], np.float32), pt, False) >= 0:
                        b['roof'] = o['roof']
        for b in _CACHE['B'] + _CACHE['SAT']:
            a = np.array(b['q'], np.float32); b['bb'] = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
        _drop_on_roads(_CACHE, GW, GH)
    return _CACHE


def drivable(t):
    """Проїжджа частина, яка має бути вільною (не під будинком)."""
    h = t.get('highway')
    return h in ROAD_W and h != 'track' and t.get('area') != 'yes' and t.get('tunnel') not in ('yes', 'building_passage') \
        and t.get('covered') != 'yes' and t.get('layer', '0') in ('0', '1', '2')


def _drop_on_roads(C, GW, GH):
    """Будинки, що стоять на проїжджій частині (похибки контурів MS/OSM, кіоски на дорозі) — прибираємо, щоб не врізатися."""
    S = 1.0 / SRC                                          # растр 1 м
    w, h = int(GW / S) + 1, int(GH / S) + 1
    road = np.zeros((h, w), np.uint8)
    for wy in C['w']:
        if not drivable(wy['t']): continue
        P = np.round(np.array(wy['pts']) / S).astype(np.int32)
        cv2.polylines(road, [P], False, 1, max(1, int(ROAD_W[wy['t']['highway']] - 1.5)))
    kept = {'B': [], 'SAT': []}; n = 0
    for key in ('B', 'SAT'):
        for b in C[key]:
            q = np.array(b['q']) / S
            x0, y0 = np.floor(q.min(0)).astype(int); x1, y1 = np.ceil(q.max(0)).astype(int) + 1
            if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h: kept[key].append(b); continue
            xa, ya = max(0, x0), max(0, y0)
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [np.round(q - [x0, y0]).astype(np.int32)], 1)
            m = m[ya - y0:min(h, y1) - y0, xa - x0:min(w, x1) - x0]
            tot = m.sum()
            frac = (road[ya:ya + m.shape[0], xa:xa + m.shape[1]][m > 0]).mean() if tot else 0
            osm = key == 'B' and b['t'].get('src') not in ('ms', 'sat', 'manual')
            if (not osm and frac > 0.12) or (osm and b['area'] < 80 and frac > 0.25) or frac > 0.6:
                n += 1; continue
            kept[key].append(b)
    C['B'], C['SAT'] = kept['B'], kept['SAT']
    C['dropped'] = n


ROAD_W = {'primary': 13, 'secondary': 11, 'tertiary': 9.5, 'residential': 7, 'unclassified': 6.5, 'living_street': 5.5,
          'primary_link': 7, 'secondary_link': 6.5, 'tertiary_link': 6.5, 'service': 4.5, 'track': 3.5}
SIDE_W = {'primary': 3.0, 'secondary': 3.0, 'tertiary': 2.6, 'residential': 2.0, 'unclassified': 1.8, 'living_street': 1.6,
          'primary_link': 2.0, 'secondary_link': 2.0, 'tertiary_link': 2.0}


def _hash(ix, iy, seed):
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (seed * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def dither(H, W, seed=1, x0=0, y0=0):
    """Шум у світових координатах: однаковий незалежно від того, яким вікном малюється карта (без швів)."""
    gx = np.arange(x0, x0 + W)[None, :]; gy = np.arange(y0, y0 + H)[:, None]
    cx = np.floor_divide(gx, 4); cy = np.floor_divide(gy, 4)
    acc = np.zeros((H, W), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += _hash(cx + dx, cy + dy, seed) * (2.0 if dx == 0 and dy == 0 else 1.0)
    n = acc / 10.0
    n = np.clip((n - 0.3) / 0.4, 0, 1)
    fine = _hash(gx + 0 * gy, gy + 0 * gx, seed + 1)
    return n, fine


# ---------------------------------------------------------------- рендер вікна
def _up(a, S):
    """Збільшити масив у S разів (кожен піксель -> S x S)."""
    return a if S == 1 else np.repeat(np.repeat(a, S, 0), S, 1)


def render(x0, y0, W, H, tree_out=None, ss=1):
    """tree_out: якщо список — дерева не малюються, а додаються туди у світових координатах (малює citybuild один раз).
    ss: надвибірка — геометрія малюється в ss разів детальніше й зменшується з усередненням (згладжені краї).
    Візерунки (плитка, розмітка, шум) і аналіз знімка — у сітці гри, тож стиль той самий, що й при ss=1."""
    D = data()
    S = int(ss)
    to = lambda p: (p[0] * K - x0, p[1] * K - y0)                 # сітка гри
    tof = lambda p: ((p[0] * K - x0) * S, (p[1] * K - y0) * S)     # детальна сітка
    WF, HF, MF = W * S, H * S, MPP / S

    def visible(pts, m=60):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return not (max(xs) < -m or min(xs) > W + m or max(ys) < -m or min(ys) > H + m)

    def near(bb, m=60):
        return not (bb[2] * K - x0 < -m or bb[0] * K - x0 > W + m or bb[3] * K - y0 < -m or bb[1] * K - y0 > H + m)
    ways = []
    for w in D['w']:
        if not near(w['bb']): continue
        pts = [to(p) for p in w['pts']]
        if visible(pts): ways.append((w, pts, [tof(p) for p in w['pts']]))
    BL = [(i, b) for i, b in enumerate(D['B'] + D['SAT']) if near(b['bb'], 40)]

    # ---------- класи землі (детальна сітка)
    G_YARD, G_GRASS, G_WATER, G_SIDE, G_ROAD, G_PATH, G_PED, G_RAIL, G_DIRT, G_PITCH, G_TRACK, G_PARK, G_PAVED, G_WOOD, G_SAND = range(1, 16)
    im = Image.new('L', (WF, HF), G_YARD); dr = ImageDraw.Draw(im)
    areas = []
    for w, pts, ptf in ways:
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
        elif na in ('beach', 'sand') or le == 'beach_resort': c = G_SAND
        if c is not None:
            a = 0.5 * abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts))))
            areas.append((a, c, ptf))
    areas.sort(key=lambda a: -a[0])
    for _, c, ptf in areas: dr.polygon(ptf, fill=c)
    for w, pts, ptf in ways:
        if w['t'].get('_inner') and len(pts) >= 4:
            of = w['t'].get('_of', '')
            if 'natural=water' in of or 'water=' in of: dr.polygon(ptf, fill=G_GRASS)
            elif 'leisure=park' in of or 'natural=wood' in of: dr.polygon(ptf, fill=G_YARD)
    # ---------- знімок: аналіз у сітці гри
    sat = np.load('build/sat_map.npy', mmap_mode='r')
    PADS = 8
    sx0, sy0 = int(math.floor(x0 / K)) - PADS, int(math.floor(y0 / K)) - PADS
    sx1, sy1 = int(math.ceil((x0 + W) / K)) + PADS, int(math.ceil((y0 + H) / K)) + PADS
    SH_, SW_ = sat.shape[:2]
    raw = np.zeros((sy1 - sy0, sx1 - sx0, 3), np.uint8)
    cy0, cx0 = max(0, sy0), max(0, sx0); cy1, cx1 = min(SH_, sy1), min(SW_, sx1)
    if cy1 > cy0 and cx1 > cx0: raw[cy0 - sy0:cy1 - sy0, cx0 - sx0:cx1 - sx0] = sat[cy0:cy1, cx0:cx1]
    raw = cv2.blur(raw, (2, 2))
    A_ = np.float32([[1 / K, 0, x0 / K + 0.5 / K - 0.5 - sx0], [0, 1 / K, y0 / K + 0.5 / K - 0.5 - sy0]])
    warp = lambda img: cv2.warpAffine(img, A_, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    sw = warp(raw)[..., ::-1].astype(np.float32)
    gray32 = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.blur(warp(np.abs(cv2.Laplacian(cv2.GaussianBlur(gray32, (3, 3), 0), cv2.CV_32F))), (2, 2))
    tex = cv2.GaussianBlur(tex, (0, 0), 1.5)
    if os.path.exists('build/sat_detail_mask.npy'):
        dm = np.load('build/sat_detail_mask.npy', mmap_mode='r')
        draw_ = np.zeros((sy1 - sy0, sx1 - sx0), np.float32)
        if cy1 > cy0 and cx1 > cx0: draw_[cy0 - sy0:cy1 - sy0, cx0 - sx0:cx1 - sx0] = dm[cy0:cy1, cx0:cx1]
        tex *= 1.0 + 1.1 * (1.0 - warp(draw_))
    sm = cv2.medianBlur(sw.astype(np.uint8), 5).astype(np.float32)
    r, g, bl = sm[..., 0], sm[..., 1], sm[..., 2]
    mx = sm.max(-1); ch = mx - sm.min(-1); exg = 2 * g - r - bl
    dark = mx < 62
    ycls = np.full((H, W), 3, np.uint8)
    ycls[exg > 6] = 1
    ycls[(r > bl + 28) & (exg <= 6) & (ch > 34) & (mx > 110)] = 2
    ycls[dark] = 0
    _bm = Image.new('L', (W, H), 0); _d = ImageDraw.Draw(_bm)
    for _, b in BL:
        P = [to(p) for p in b['q']]
        if visible(P, 30): _d.polygon(P, fill=1)
    for w, pts, ptf in ways:
        hw = w['t'].get('highway')
        if hw in ROAD_W and w['t'].get('area') != 'yes': _d.line(pts, fill=1, width=int(ROAD_W[hw] / MPP) + 2)
    ycls[cv2.dilate(np.array(_bm, np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
    if (ycls == 0).any():
        _, (iy, ix) = ndi.distance_transform_edt(ycls == 0, return_indices=True); ycls = ycls[iy, ix]
    oh = np.stack([cv2.blur((ycls == k).astype(np.float32), (9, 9)) for k in (1, 2, 3)])
    ycls = _up(oh.argmax(0).astype(np.uint8) + 1, S)
    cls = np.array(im)
    yard = cls == G_YARD
    cls[yard & (ycls == 1)] = G_GRASS; cls[yard & (ycls == 2)] = G_DIRT; cls[yard & (ycls == 3)] = G_PAVED

    # лінійні: вода, рейки, тротуари, доріжки, дороги
    im = Image.fromarray(cls); dr = ImageDraw.Draw(im)
    def line(pts, wm, fill, d=dr):
        wp = max(1.0 * S, wm / MF); d.line(pts, fill=fill, width=int(round(wp)))
        rr = wp / 2
        for x, y in pts: d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill)
    WWW = {'river': 20, 'canal': 10, 'stream': 4, 'ditch': 2, 'drain': 2}
    rails, drive, foot = [], [], []
    for w, pts, ptf in ways:
        t = w['t']
        if t.get('waterway') in WWW and t.get('tunnel') not in ('yes', 'culvert'): line(ptf, WWW[t['waterway']], G_WATER)
        if t.get('railway') in ('rail', 'siding', 'narrow_gauge') and t.get('tunnel') != 'yes':
            line(ptf, 4.0, G_RAIL); rails.append(ptf)
        h = t.get('highway')
        if h and t.get('tunnel') != 'yes' and t.get('area') != 'yes':
            if h in ROAD_W: drive.append((h, ptf, w))
            elif h in ('footway', 'path', 'cycleway', 'steps', 'pedestrian', 'bridleway'): foot.append((h, ptf, w))
    for h, pts, w in drive:
        if h in SIDE_W and w['t'].get('sidewalk') != 'no': line(pts, ROAD_W[h] + 2 * SIDE_W[h], G_SIDE)
    for h, pts, w in foot:
        line(pts, 8 if h == 'pedestrian' else 2.0, G_PED if h == 'pedestrian' else G_PATH)
    for h, pts, w in drive: line(pts, ROAD_W[h], G_ROAD)
    cls = np.array(im)

    # ---------- текстури землі (візерунки — у сітці гри)
    img = np.zeros((HF, WF, 3), np.uint8)
    blob, fine = dither(H, W, 7, int(round(x0)), int(round(y0)))
    blob, fine = _up(blob, S), _up(fine, S)
    if S > 1: blob = cv2.GaussianBlur(blob, (0, 0), 1.6 * S)         # плавний шум замість квадратиків 4x4
    smf = _up(sm, S)
    yyf, xxf = np.mgrid[0:HF, 0:WF]
    yy = yyf // S + int(round(y0)); xx = xxf // S + int(round(x0))
    def pick(ramp, t):
        if S == 1:
            idx = np.clip((t * len(ramp)).astype(int), 0, len(ramp) - 1); return np.array(ramp, np.uint8)[idx]
        R_ = np.array(ramp, np.float32); n = len(ramp)                  # плавний перехід між відтінками
        u = np.clip(t, 0, 1) * (n - 1); i = np.clip(np.floor(u).astype(int), 0, n - 2) if n > 1 else np.zeros_like(u, int)
        f = (u - i)[..., None] if n > 1 else 0
        return (R_[i] * (1 - f) + R_[np.minimum(i + 1, n - 1)] * f).astype(np.uint8)
    def put(cid, arr):
        m = cls == cid; img[m] = arr[m]
    spk = 0.25 if S == 1 else 0.12
    t_grass = 0.25 + 0.5 * blob + spk * (fine > 0.86) - 0.6 * spk * (fine < 0.08)
    put(G_GRASS, pick(GRASS[1:], t_grass)); put(G_WOOD, pick(GRASS[:3], t_grass))
    dry = (cls == G_GRASS) & (smf[..., 0] > smf[..., 1] + 4)
    img[dry] = pick(DRY, blob)[dry]
    put(G_DIRT, pick(DIRT, 0.3 + 0.5 * blob + 0.2 * (fine > 0.9)))
    put(G_PAVED, pick(PAVED, 0.35 + 0.3 * blob + 0.2 * (fine > 0.92)))
    tile4 = ((xx % 4 == 0) | (yy % 4 == 0))
    put(G_SIDE, np.where(tile4[..., None], np.array(SIDE[1], np.uint8), np.array(SIDE[2], np.uint8)))
    brick = (yy % 3 == 0) | (((xx + (yy // 3) % 2 * 3) % 6) == 0)
    put(G_PED, np.where(brick[..., None], np.array(SIDE[0], np.uint8), np.array(SIDE[2], np.uint8)))
    put(G_PATH, np.full((HF, WF, 3), SIDE[1], np.uint8))
    asph = pick(ASPH, 0.35 + 0.35 * blob + 0.3 * (fine > 0.93))
    put(G_ROAD, asph); put(G_PARK, pick(ASPH[1:], 0.4 + 0.4 * blob))
    rip = (np.sin(xx * 0.5 + yy * 0.15 + blob * 4) > 0.85)
    put(G_WATER, np.where(rip[..., None], np.array(WATER[2], np.uint8), pick(WATER[:2], 0.3 + 0.5 * blob)))
    put(G_RAIL, pick(BALLAST, fine))
    put(G_PITCH, np.where(((xx // 8) % 2 == 0)[..., None], np.array(PITCH[0], np.uint8), np.array(PITCH[1], np.uint8)))
    put(G_TRACK, np.full((HF, WF, 3), TRACK[1], np.uint8))
    put(G_SAND, pick([(206, 186, 140), (218, 200, 154), (228, 212, 168)], 0.3 + 0.4 * blob + 0.3 * (fine > 0.9)))
    put(G_YARD, pick(PAVED, 0.5 + 0.0 * blob))
    road = cls == G_ROAD
    walk = np.isin(cls, (G_SIDE, G_PED))
    k3 = np.ones((2 * S + 1, 2 * S + 1), np.uint8)
    img[walk & cv2.dilate(road.astype(np.uint8), k3).astype(bool)] = CURB
    soft = np.isin(cls, (G_GRASS, G_WOOD, G_DIRT))
    img[walk & cv2.dilate(soft.astype(np.uint8), k3).astype(bool)] = SIDE[0]
    # рейки
    ov = Image.new('L', (WF, HF), 0); do = ImageDraw.Draw(ov)
    for pts in rails:
        for i in range(len(pts) - 1):
            (xa, ya), (xb, yb) = pts[i], pts[i + 1]; L = math.hypot(xb - xa, yb - ya)
            if L < 1: continue
            ux, uy = (xb - xa) / L, (yb - ya) / L; s_ = 0.0
            while s_ < L:
                cx, cy = xa + ux * s_, ya + uy * s_
                do.line([(cx + uy * 3.5 * S, cy - ux * 3.5 * S), (cx - uy * 3.5 * S, cy + ux * 3.5 * S)], fill=1, width=S); s_ += 2.5 * S
        for off in (-1.6 * S, 1.6 * S):
            sh = []
            for i in range(len(pts)):
                a = pts[max(i - 1, 0)]; b2 = pts[min(i + 1, len(pts) - 1)]
                L = math.hypot(b2[0] - a[0], b2[1] - a[1]) or 1
                sh.append((pts[i][0] - (b2[1] - a[1]) / L * off, pts[i][1] + (b2[0] - a[0]) / L * off))
            do.line(sh, fill=2, width=S)
    ova = np.array(ov); img[ova == 1] = SLEEPER; img[ova == 2] = RAIL
    # розмітка, переходи
    mk = Image.new('L', (WF, HF), 0); dm = ImageDraw.Draw(mk)
    for h, pts, w in drive:
        if h not in ('primary', 'secondary', 'tertiary') or w['t'].get('oneway') == 'yes': continue
        s0 = 0.0
        for i in range(len(pts) - 1):
            (xa, ya), (xb, yb) = pts[i], pts[i + 1]; L = math.hypot(xb - xa, yb - ya); s_ = 0.0
            while s_ < L:
                ph = (s0 + s_) % (10 * S)
                if ph < 5 * S:
                    e = min(s_ + 5 * S - ph, L)
                    dm.line([(xa + (xb - xa) * s_ / L, ya + (yb - ya) * s_ / L), (xa + (xb - xa) * e / L, ya + (yb - ya) * e / L)], fill=1, width=S)
                    s_ = e
                else: s_ += 10 * S - ph
            s0 += L
    node_road = {}
    for h, pts, w in drive:
        for k, nid in enumerate(w['nds']): node_road.setdefault(nid, (h, pts, k))
    for nid, t in D['n'].items():
        if t.get('highway') != 'crossing' or nid not in node_road: continue
        h, pts, k = node_road[nid]
        a = pts[max(k - 1, 0)]; b2 = pts[min(k + 1, len(pts) - 1)]
        L = math.hypot(b2[0] - a[0], b2[1] - a[1]) or 1; ux, uy = (b2[0] - a[0]) / L, (b2[1] - a[1]) / L
        cx, cy = pts[k]; hw = ROAD_W[h] / MF / 2; v = -hw + S
        while v < hw - 0.5 * S:
            dm.line([(cx - uy * v - ux * 3 * S, cy + ux * v - uy * 3 * S), (cx - uy * v + ux * 3 * S, cy + ux * v + uy * 3 * S)], fill=1, width=2 * S); v += 3 * S
    img[(np.array(mk) > 0) & road] = MARK
    # розмітка паркувань
    for w, pts, ptf in ways:
        if not (w['closed'] and w['t'].get('amenity') == 'parking' and len(pts) >= 4): continue
        P = np.array(pts, np.float32)
        m = np.zeros((HF, WF), np.uint8); cv2.fillPoly(m, [np.round(np.array(ptf)).astype(np.int32)], 1)
        m = cv2.erode(m, np.ones((4 * S + 1, 4 * S + 1), np.uint8)).astype(bool) & (cls == G_PARK)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(P); a = math.radians(ang)
        ux, uy = (math.cos(a), math.sin(a)) if rw >= rh else (-math.sin(a), math.cos(a))
        xl = xx - int(round(x0)); yl = yy - int(round(y0))
        along = xl * ux + yl * uy; across = -xl * uy + yl * ux
        c_across = -cx * uy + cy * ux
        band = (np.abs(((across - c_across) % 22) - 11) > 2)
        stripe = (np.floor(along) % 5 == 0) & band
        img[m & stripe] = (178, 180, 182)
    # бігова доріжка
    stad = [np.array(p, np.float32) for w, p, _ in ways if w['closed'] and w['t'].get('leisure') == 'stadium']
    for w, pts, ptf in ways:
        if not (w['closed'] and w['t'].get('leisure') == 'pitch' and len(pts) >= 4): continue
        P = np.array(pts, np.float32); c = P.mean(0)
        if not any(cv2.pointPolygonTest(S_, (float(c[0]), float(c[1])), False) >= 0 for S_ in stad): continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(np.array(ptf, np.float32))
        L, Sx = max(rw, rh), min(rw, rh)
        a = math.radians(ang) if rw >= rh else math.radians(ang + 90)
        ux, uy = math.cos(a), math.sin(a); vx, vy = -uy, ux
        m = np.zeros((HF, WF), np.uint8)
        half = L / 2 - Sx * 0.22
        R_in = math.hypot(L / 2 - half, Sx / 2) + 2 * S
        for k, off in enumerate((9.0 / MF, 0.0)):
            R = R_in + off
            poly = []
            for t in np.linspace(-90, 90, 24):
                tr = math.radians(t); poly.append((cx + ux * (half + R * math.cos(tr)) + vx * R * math.sin(tr), cy + uy * (half + R * math.cos(tr)) + vy * R * math.sin(tr)))
            for t in np.linspace(90, 270, 24):
                tr = math.radians(t); poly.append((cx + ux * (-half + R * math.cos(tr)) + vx * R * math.sin(tr), cy + uy * (-half + R * math.cos(tr)) + vy * R * math.sin(tr)))
            cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 1 if k == 0 else 2)
        trk = m == 1
        img[trk] = TRACK[1]
        dd = cv2.distanceTransform((m == 1).astype(np.uint8), cv2.DIST_L2, 3) / S
        img[trk & (np.round(dd) % 3 == 0) & (dd > 1)] = (214, 170, 150)
        img[m == 2] = np.where(((xx // 8) % 2 == 0)[..., None], np.array(PITCH[0], np.uint8), np.array(PITCH[1], np.uint8))[m == 2]
        cls[m > 0] = G_PITCH
    for w, pts, ptf in ways:
        if w['closed'] and w['t'].get('leisure') == 'pitch' and len(pts) >= 4:
            P = np.array(ptf, np.float32); rect = cv2.minAreaRect(P); bx = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(img, [bx], True, LINE, S)
            (cx, cy), (rw, rh), ang = rect; a = math.radians(ang)
            if rw >= rh: ux, uy = -math.sin(a), math.cos(a); half = rh / 2
            else: ux, uy = math.cos(a), math.sin(a); half = rw / 2
            cv2.line(img, (int(cx - ux * half), int(cy - uy * half)), (int(cx + ux * half), int(cy + uy * half)), LINE, S)
            cv2.circle(img, (int(cx), int(cy)), max(3 * S, int(min(rw, rh) * 0.12)), LINE, S)
        if w['closed'] and w['t'].get('leisure') == 'track' and len(pts) >= 4:
            P = np.round(np.array(ptf)).astype(np.int32)
            m = np.zeros((HF, WF), np.uint8); cv2.fillPoly(m, [P], 1)
            d = cv2.distanceTransform(m, cv2.DIST_L2, 3) / S
            lane = (m > 0) & (np.round(d) % 3 == 0) & (d > 1)
            img[lane] = TRACK[0]

    # ---------- будинки
    blds = []
    for i, b in BL:
        pts = [to(p) for p in b['q']]
        if not visible(pts, 30): continue
        q = b['q']
        if S > 1 and len(q) > 6:                        # дрібні сходинки контуру (<0,4 м) прибрати
            Q = np.array(q, np.float32) * K
            Q2 = cv2.approxPolyDP(Q.reshape(-1, 1, 2), 0.8, True).reshape(-1, 2) / K
            if len(Q2) >= 3: q = [tuple(p) for p in Q2]
            pts = [to(p) for p in q]
        blds.append((i, b, pts, [tof(p) for p in q]))
    bmaskf = np.zeros((HF, WF), np.uint8)
    for i, b, pts, ptf in blds: cv2.fillPoly(bmaskf, [np.round(np.array(ptf)).astype(np.int32)], 1)
    bmaskf = bmaskf.astype(bool)
    bmask = bmaskf[S // 2::S, S // 2::S]
    clsb = cls[S // 2::S, S // 2::S]

    # ---------- дерева зі знімка (у сітці гри)
    green = exg > 8; autumn = (r > bl + 30) & (ch > 34)
    canopy = (tex > 4.0) & (green | autumn | ((ch > 18) & (mx < 120) & (tex > 7))) & ~(mx < 40)
    park = np.zeros((H, W), np.uint8)
    for w, pts, ptf in ways:
        if w['closed'] and (w['t'].get('leisure') == 'park' or w['t'].get('natural') == 'wood' or w['t'].get('landuse') == 'forest'):
            cv2.fillPoly(park, [np.round(np.array(pts)).astype(np.int32)], 1)
    park = park.astype(bool) & ~np.isin(clsb, (G_PATH, G_PED, G_ROAD, G_SIDE))
    canopy |= park & ((tex > 3.0) | (mx < 70)) & (exg > -8)
    canopy &= ~cv2.dilate(bmask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    canopy &= ~np.isin(clsb, (G_ROAD, G_RAIL, G_WATER, G_PITCH, G_TRACK, G_SAND))
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

    # ---------- тіні (на південний схід), м'які краї
    shadow = np.zeros((HF, WF), np.uint8)
    for i, b, pts, ptf in blds:
        s_ = int(np.clip(b['h'] * 0.22, 1, 7)) * S
        P = np.round(np.array(ptf)).astype(np.int32)
        for k in range(1, s_ + 1, max(1, S // 2)): cv2.fillPoly(shadow, [P + [k, k]], 1)
    for x, y, rr, _ in trees: cv2.circle(shadow, (int((x + 2.5) * S), int((y + 2.5) * S)), int(rr * S), 1, -1)
    shf = shadow.astype(np.float32)
    if S > 1:
        shf = cv2.GaussianBlur(shf, (0, 0), 1.0 * S)
        shf = np.maximum(shf, 0.55 * cv2.GaussianBlur(bmaskf.astype(np.float32), (0, 0), 1.5 * S))   # затінення біля стін
    shf[bmaskf] = 0
    img = (img.astype(np.float32) * (1 - (0.3 if S == 1 else 0.34) * shf[..., None])).astype(np.uint8)

    # ---------- пляж
    sand = cls == G_SAND
    if sand.any():
        gx0, gy0 = int(round(x0)), int(round(y0))
        UMB = [(214, 64, 52), (244, 196, 52), (52, 120, 200), (238, 238, 232)]
        for yy_ in range(-(gy0 % 16) + 8, H, 16):
            for xx_ in range(-(gx0 % 18) + 9, W, 18):
                if not (0 <= yy_ < H and 0 <= xx_ < W) or not sand[yy_ * S, xx_ * S]: continue
                hsh = ((xx_ + gx0) * 73856093 ^ (yy_ + gy0) * 19349663) & 0xFFFF
                if hsh % 3 == 0: continue
                if not sand[max(0, yy_ - 5) * S:(yy_ + 6) * S, max(0, xx_ - 5) * S:(xx_ + 6) * S].all(): continue
                c = UMB[hsh % 4]
                cv2.rectangle(img, ((xx_ + 3) * S, (yy_ + 1) * S), ((xx_ + 7) * S, (yy_ + 2) * S + S - 1), (236, 232, 220), -1)
                cv2.circle(img, (xx_ * S + S // 2, yy_ * S + S // 2), 4 * S, OUT, -1); cv2.circle(img, (xx_ * S + S // 2, yy_ * S + S // 2), 3 * S, c, -1)
    # ---------- дахи
    img_ground = img.copy()
    for i, b, pts, ptf in sorted(blds, key=lambda z: z[1]['h']):
        draw_roof(img, b, ptf, random.Random(i * 31 + 7), S)
    # проїжджа частина завжди видна: дах, що трохи наліз на дорогу, обрізається (крім арок)
    rm = Image.new('L', (WF, HF), 0); drm = ImageDraw.Draw(rm)
    for w, pts, ptf in ways:
        if drivable(w['t']): drm.line(ptf, fill=1, width=max(1, int(ROAD_W[w['t']['highway']] / MF)))      # уся ширина проїжджої частини
    rmask = (np.array(rm) > 0) & (cls == G_ROAD)
    img[rmask] = img_ground[rmask]

    # ---------- дерева
    if tree_out is None:
        for k, (x, y, rr, kind) in enumerate(sorted(trees, key=lambda t: t[1])):
            draw_tree(img, x * S + S // 2, y * S + S // 2, rr * S, kind, random.Random(int(x + x0) * 7919 + int(y + y0)), S)
    if S > 1:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        cls = clsb.copy()
    if tree_out is not None:
        tree_out.extend((int(x + x0), int(y + y0), rr, kind) for x, y, rr, kind in trees)
    return img, cls


def draw_roof(img, b, pts, rs, ss=1):
    H, W = img.shape[:2]
    S_ = ss
    P = np.array(pts, np.float32)
    bx0, by0 = np.floor(P.min(0)).astype(int) - 3 * S_; bx1, by1 = np.ceil(P.max(0)).astype(int) + 4 * S_
    if bx1 < 0 or by1 < 0 or bx0 >= W or by0 >= H: return
    w, h = bx1 - bx0, by1 - by0
    m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(P - [bx0, by0]).astype(np.int32)], 1)
    F = m.astype(bool)
    if F.sum() < 4 * S_ * S_: return
    area = b['area']; t = b['t']; bt = t.get('building', 'yes')
    per = float(np.sum(np.linalg.norm(np.diff(np.array(b['q']), axis=0), axis=1))) * SRC
    circ = 4 * math.pi * area / max(per, 1) ** 2
    kind = roof_kind(b.get('roof'))
    S = [np.array(c, np.uint8) for c in shades(ROOFS[kind])]
    out = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3) / S_            # у пікселях гри
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
        dd = cv2.GaussianBlur(d, (0, 0), 0.8 * S_)
        gy, gx = np.gradient(dd)
        nrm = (np.hypot(gx, gy) + 1e-6) * S_
        lit = (-gx - gy) * S_ / nrm / 1.414
        ridge = (nrm < 0.25) & (d > 1.5)
        if S_ == 1:
            out[:] = S[2]
            out[lit > 0.35] = S[3]; out[lit < -0.35] = S[1]
            out[ridge] = S[4]
        else:                                           # плавне світло скатів, м'який гребінь
            Sf = [x.astype(np.float32) for x in S]
            lp = np.clip(lit, 0, 1)[..., None]; ln = np.clip(-lit, 0, 1)[..., None]
            col = Sf[2] + (Sf[3] - Sf[2]) * lp * 1.3 + (Sf[1] - Sf[2]) * ln * 1.3
            rg = (np.clip((0.4 - nrm) / 0.3, 0, 1) * (d > 1.2))[..., None]
            col = col * (1 - rg * 0.6) + Sf[4] * rg * 0.6
            col *= (0.88 + 0.12 * np.clip(d / 2.5, 0, 1))[..., None]          # темніше до карниза
            out[:] = np.clip(col, 0, 255).astype(np.uint8)
    elif bt in ('garage', 'garages', 'shed', 'industrial', 'warehouse', 'roof', 'service') or kind == 'metal':
        rect = cv2.minAreaRect(P); (rw, rh), a = rect[1], math.radians(rect[2])
        ux, uy = (math.cos(a), math.sin(a)) if rw < rh else (-math.sin(a), math.cos(a))
        st = ((xx * ux + yy * uy) // (2 * S_)) % 2 == 0
        out[:] = np.where(st[..., None], S[2], S[3])
        out[(d >= 1) & (d < 2)] = S[1]
    else:
        out[:] = S[2]
        out[(d >= 1) & (d < 2.2)] = S[3]
        out[(d >= 2.2) & (d < 3.2)] = S[1]
        ys, xs = np.nonzero(d > 6)
        for _ in range(min(len(xs) // (400 * S_ * S_), 5)):
            k = rs.randrange(len(xs)); vx, vy = xs[k], ys[k]
            out[vy:vy + 3 * S_, vx:vx + 4 * S_] = S[3]; out[vy + 3 * S_:vy + 4 * S_, vx + S_:vx + 5 * S_] = S[1]
    out[F & (d < 1.01)] = S[0] if circ > 0.9 else out[F & (d < 1.01)]
    ko = 2 * S_ + 1 if S_ == 1 else S_ + 1                # обвідка: 1 пікс гри; при надвибірці ~0,5 пікс
    ol = cv2.dilate(m, np.ones((ko, ko), np.uint8)).astype(bool) & ~F
    dst = img[max(0, by0):min(H, by1), max(0, bx0):min(W, bx1)]
    oy, ox = max(0, by0) - by0, max(0, bx0) - bx0
    Fs = F[oy:oy + dst.shape[0], ox:ox + dst.shape[1]]; Os = ol[oy:oy + dst.shape[0], ox:ox + dst.shape[1]]
    dst[Fs] = out[oy:oy + dst.shape[0], ox:ox + dst.shape[1]][Fs]
    dst[Os] = OUT if S_ == 1 else (S[0].astype(np.float32) * 0.55).astype(np.uint8)


def draw_tree(img, x, y, rr, kind, rs, ss=1):
    """x, y, rr — у сітці img (при ss>1 img детальний, rr уже помножений на ss)."""
    H, W = img.shape[:2]
    R = int(math.ceil(rr)) + 2 * ss
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    m = xx ** 2 + yy ** 2 <= (rr * 0.82) ** 2
    for _ in range(6):
        a = rs.random() * 6.283; dd = rr * (0.3 + rs.random() * 0.3); lr = rr * (0.35 + rs.random() * 0.2)
        m |= (xx - math.cos(a) * dd) ** 2 + (yy - math.sin(a) * dd) ** 2 <= lr ** 2
    C = [np.array(c, np.uint8) for c in TREE[kind]]
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3) / ss
    lit = (-xx - yy) / (rr + 1e-3)
    o = np.zeros(m.shape + (3,), np.uint8); o[:] = C[2]
    if ss == 1:
        o[lit < -0.25] = C[1]; o[(lit > 0.35) & (dist > 1.5)] = C[3]
    else:                                                 # плавна світлотінь, край крони темніший
        Cf = [c.astype(np.float32) for c in C]
        lp = np.clip(lit, 0, 1)[..., None]; ln = np.clip(-lit, 0, 1)[..., None]
        col = Cf[2] + (Cf[3] - Cf[2]) * lp * 1.2 + (Cf[1] - Cf[2]) * ln * 1.2
        col = col * (0.82 + 0.18 * np.clip(dist / 2.0, 0, 1))[..., None]
        o[:] = np.clip(col, 0, 255).astype(np.uint8)
    n0 = (2 * R) // ss + 2
    spk = _up(np.random.RandomState(rs.randrange(1 << 30)).rand(n0, n0) < 0.06, ss)[:2 * R + 1, :2 * R + 1] & (lit > 0)
    o[spk] = C[3]
    if ss == 1: o[m & (dist < 1.01)] = C[0]
    ko = 2 * ss + 1 if ss == 1 else ss + 1
    ol = cv2.dilate(m.astype(np.uint8), np.ones((ko, ko), np.uint8)).astype(bool) & ~m
    o[ol] = OUT if ss == 1 else (C[0].astype(np.float32) * 0.6).astype(np.uint8)
    x0, y0 = x - R, y - R
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(W, x0 + 2 * R + 1), min(H, y0 + 2 * R + 1)
    if ax1 <= ax0 or ay1 <= ay0: return
    sl = (slice(ay0 - y0, ay1 - y0), slice(ax0 - x0, ax1 - x0))
    msk = (m | ol)[sl]
    dst = img[ay0:ay1, ax0:ax1]
    dst[msk] = o[sl][msk]


def draw_tree_aa(img, x, y, rr, kind, rs, ss=4):
    """Дерево зі згладженими краями прямо на карті сітки гри: малюється в ss разів детальніше й накладається з покриттям."""
    R = int(math.ceil(rr)) + 3
    n = 2 * R + 1
    KEY = np.array((255, 0, 255), np.uint8)
    big = np.empty((n * ss, n * ss, 3), np.uint8); big[:] = KEY
    c = R * ss + ss // 2
    draw_tree(big, c, c, rr * ss, kind, rs, ss)
    m = np.any(big != KEY, -1).astype(np.float32)
    a = cv2.resize(m, (n, n), interpolation=cv2.INTER_AREA)[..., None]
    col = cv2.resize(big.astype(np.float32) * m[..., None], (n, n), interpolation=cv2.INTER_AREA) / np.maximum(a, 1e-4)
    H, W = img.shape[:2]
    x0, y0 = x - R, y - R
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(W, x0 + n), min(H, y0 + n)
    if ax1 <= ax0 or ay1 <= ay0: return
    sl = (slice(ay0 - y0, ay1 - y0), slice(ax0 - x0, ax1 - x0))
    dst = img[ay0:ay1, ax0:ax1].astype(np.float32)
    img[ay0:ay1, ax0:ax1] = np.clip(dst * (1 - a[sl]) + col[sl] * a[sl], 0, 255).astype(np.uint8)


if __name__ == '__main__':
    if sys.argv[1] == 'sample':
        cx, cy, w, h = map(int, sys.argv[2:6])
        img, _ = render(cx - w // 2, cy - h // 2, w, h, ss=int(os.environ.get('SS', '1')))
        cv2.imwrite(sys.argv[6], img[..., ::-1])
