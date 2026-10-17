"""Етап 1: впізнаване місто. OSM (геометрія) + супутник (дахи, дерева, двори, висоти) -> індексована піксель-арт карта.
Вхід: data/osm_v2, build/sat_map.npy (+json), build/buildings.json, build/bounds.npz
Вихід: build/map.npy, build/solid.npy, build/mapdata.json, build/preview_*.png
usage: python3 -I tools/citygen.py
"""
import math, glob, sys, os, json, random
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pal import PAL, R, dark_lut
import satclass
import mapgen as MG                      # тексти/скорочення назв, штампи дерев, ширини доріг

OUT = 'build'
META = json.load(open('build/sat_map.json'))
M = META['mpp']
X0, Y0 = META['x0'], META['y0']
CS = 64
W, H = META['W'] // CS * CS, META['H'] // CS * CS
LAT0, LON0 = 49.422, 26.995
MLAT = 111320.0
MLON = 111320.0 * math.cos(math.radians(LAT0))
WIDEN = 1.15

(C_YARD, C_GRASS, C_WOOD, C_WATER, C_SIDE, C_ROAD, C_PATH, C_PED, C_CONC, C_RAIL, C_DIRT, C_PITCH, C_PARKING,
 C_TRACK, C_PAVED, C_ASPHY) = range(1, 17)
SOFT = (C_YARD, C_GRASS, C_WOOD, C_DIRT, C_CONC)

rng = np.random.RandomState(1234)
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0
DARK = np.array(dark_lut(), np.uint8)


def px(lat, lon):
    return (((lon - LON0) * MLON - X0) / M, (-(lat - LAT0) * MLAT - Y0) / M)


def load(d):
    nodes, ntags, ways = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(d, 't*.osm'))):
        root = ET.parse(f).getroot()
        for n in root.iter('node'):
            nodes[n.get('id')] = (float(n.get('lat')), float(n.get('lon')))
            t = {x.get('k'): x.get('v') for x in n.iter('tag')}
            if t: ntags[n.get('id')] = t
        for w in root.iter('way'):
            ways[w.get('id')] = ([nd.get('ref') for nd in w.iter('nd')], {x.get('k'): x.get('v') for x in w.iter('tag')})
    out = []
    for wid, (nds, tags) in ways.items():
        ids = [i for i in nds if i in nodes]
        pts = [px(*nodes[i]) for i in ids]
        if len(pts) >= 2:
            out.append(dict(id=wid, nds=ids, pts=pts, t=tags, closed=(nds[0] == nds[-1] and len(nds) > 3)))
    return nodes, ntags, out


def road_width_m(t):
    return MG.road_width_m(t) / MG.WIDEN * WIDEN


def draw_line(dr, pts, wpx, fill):
    MG.draw_line(dr, pts, wpx, fill)


def smooth_noise(sigma, down):
    h, w = H // down + 2, W // down + 2
    n = ndi.gaussian_filter(rng.randn(h, w).astype(np.float32), sigma / down)
    n /= (n.std() + 1e-6)
    return ndi.zoom(n, down, order=1)[:H, :W].astype(np.float32)


def quant(T, ramp, bay, dith=0.85):
    ids = np.array(R[ramp], np.uint8)
    i = np.clip(np.floor(T + (bay - 0.5) * dith + 0.5), 0, len(ids) - 1).astype(np.int32)
    return ids[i]


# ------------------------------------------------------------------ 1. класи з OSM
def osm_classes(ways):
    im = Image.new('L', (W, H), C_YARD); dc = ImageDraw.Draw(im)
    areas = []
    for w in ways:
        if not w['closed']: continue
        t = w['t']; c = None
        lu, le, na, am = t.get('landuse'), t.get('leisure'), t.get('natural'), t.get('amenity')
        if lu in ('retail', 'commercial', 'industrial', 'garages', 'railway'): c = C_CONC
        elif lu in ('construction', 'brownfield'): c = C_DIRT
        elif lu in ('grass', 'meadow', 'recreation_ground', 'village_green', 'flowerbed', 'cemetery'): c = C_GRASS
        elif le in ('park', 'garden', 'playground', 'dog_park'): c = C_GRASS
        elif le == 'track': c = C_TRACK
        elif le == 'pitch': c = C_PITCH
        elif le in ('stadium', 'sports_centre'): c = C_GRASS
        elif na in ('wood', 'scrub') or lu == 'forest': c = C_WOOD
        elif na in ('grassland', 'heath'): c = C_GRASS
        elif na == 'water' or lu in ('reservoir', 'basin') or le == 'swimming_pool': c = C_WATER
        elif am == 'parking' and t.get('parking') not in ('underground', 'multi-storey'): c = C_PARKING
        elif t.get('highway') == 'pedestrian' and t.get('area') == 'yes': c = C_PED
        elif t.get('place') == 'square' or t.get('area:highway') == 'pedestrian': c = C_PED
        elif t.get('railway') == 'platform': c = C_PAVED
        if c is not None:
            areas.append((MG.poly_area(w['pts']), c, w['pts']))
    areas.sort(key=lambda a: -a[0])
    for _, c, pts in areas:
        dc.polygon(pts, fill=c)
    WW = {'river': 22, 'canal': 10, 'stream': 4, 'ditch': 2, 'drain': 2}
    for w in ways:
        ww = w['t'].get('waterway')
        if ww in WW and w['t'].get('tunnel') not in ('yes', 'culvert'):
            draw_line(dc, w['pts'], WW[ww] / M, C_WATER)
    rails = []
    for w in ways:
        rw = w['t'].get('railway')
        if rw in ('rail', 'narrow_gauge', 'disused', 'siding') and w['t'].get('tunnel') != 'yes':
            draw_line(dc, w['pts'], 4.2 / M, C_RAIL); rails.append(w['pts'])
    drive, foot = [], []
    for w in ways:
        t = w['t']; h = t.get('highway')
        if not h or t.get('tunnel') == 'yes' or t.get('area') == 'yes': continue
        if h in MG.ROADW: drive.append(w)
        elif h in ('footway', 'path', 'cycleway', 'steps', 'pedestrian', 'bridleway', 'corridor'): foot.append(w)
    for w in drive:
        h = w['t']['highway']
        if h in MG.SIDEW and w['t'].get('sidewalk') != 'no':
            draw_line(dc, w['pts'], (road_width_m(w['t']) + 2 * MG.SIDEW[h] * WIDEN) / M, C_SIDE)
    for w in foot:
        h = w['t']['highway']
        if h == 'pedestrian': draw_line(dc, w['pts'], 9 / M, C_PED)
        elif w['t'].get('footway') in ('sidewalk', 'crossing'): draw_line(dc, w['pts'], 2.6 / M, C_SIDE)
        else: draw_line(dc, w['pts'], 2.2 / M, C_PATH)
    for w in drive:
        draw_line(dc, w['pts'], road_width_m(w['t']) / M, C_ROAD)
    return np.array(im), drive, foot, rails


# ------------------------------------------------------------------ 2. супутник: двори й дерева
def satellite_layers(cls, bmask):
    sat = np.load('build/sat_map.npy', mmap_mode='r')
    rgb, tex = satclass.features(sat)
    F = satclass.F
    h, w = H // F, W // F
    rgb, tex = rgb[:h, :w], tex[:h, :w]
    c1 = cls[1::F, 1::F][:h, :w]
    b1 = bmask[1::F, 1::F][:h, :w]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1); mn = rgb.min(-1); ch = mx - mn; lum = rgb.mean(-1)
    t = ndi.uniform_filter(tex, 3)
    exg = 2 * g - r - b
    dark = mx < 56
    shadow = dark | ((lum < 82) & (ch < 30))
    # --- дерева
    green = exg >= 12
    autumn = (r > b + 34) & (ch > 38)
    canopy = ((t > 8.5) & (green | autumn) & ~dark) | ((t > 11) & (ch > 22) & (mx < 110) & ~dark)
    bad = np.isin(c1, (C_ROAD, C_RAIL, C_WATER, C_PITCH, C_TRACK, C_PARKING, C_PED)) | ndi.binary_dilation(b1, iterations=1)
    canopy &= ~bad
    canopy = ndi.binary_closing(canopy, iterations=1)
    canopy = ndi.binary_opening(canopy, iterations=1)
    lab, n = ndi.label(canopy)
    sz = ndi.sum(canopy, lab, np.arange(1, n + 1))
    keep = np.r_[False, sz >= 9]
    canopy = keep[lab]
    canopy &= ~bad
    # --- двори
    g1 = np.full((h, w), 0, np.uint8)            # 0 невідомо
    paved = ch < 22
    g1[paved & (lum >= 112)] = C_PAVED
    g1[paved & (lum < 112)] = C_ASPHY
    g1[(r > b + 12) & (ch >= 22) & (exg < 12)] = C_DIRT
    g1[exg >= 12] = C_GRASS
    g1[shadow | canopy] = 0
    cnt = np.stack([ndi.uniform_filter((g1 == k).astype(np.float32), 7) for k in (0, C_PAVED, C_ASPHY, C_DIRT, C_GRASS)])
    known = cnt[1:].sum(0)
    best = np.array([C_PAVED, C_ASPHY, C_DIRT, C_GRASS], np.uint8)[cnt[1:].argmax(0)]
    best[known < 0.3] = 0
    # невідоме (тінь, крони) — найближчим відомим класом у межах ~15 м
    unk = best == 0
    dd, (iy, ix) = ndi.distance_transform_edt(unk, return_indices=True)
    best = np.where(unk & (dd <= 15), best[iy, ix], best)
    del dd, iy, ix
    # тон трави (жовтизна) зі знімка
    yel = ndi.uniform_filter(np.clip((r - g + 10) / 30.0, 0, 1), 9)
    gl = ndi.uniform_filter(np.clip((lum - 50) / 40.0, 0, 2.2), 7)
    return dict(F=F, h=h, w=w, rgb=rgb, canopy=canopy, ground=best, yel=yel.astype(np.float32), glum=gl.astype(np.float32),
                tex=t)


def place_trees(L, solid_like):
    F = L['F']; canopy = L['canopy']; rgb = L['rgb']
    d = ndi.distance_transform_edt(canopy).astype(np.float32)        # у кроках ~1 м
    pk = (d == ndi.maximum_filter(d, size=5)) & (d >= 1.4)
    ys, xs = np.nonzero(pk)
    order = np.argsort(-d[ys, xs])
    ys, xs = ys[order], xs[order]
    trees = []
    acc = []
    tr = None
    for y, x in zip(ys, xs):
        rm = float(min(d[y, x] * 1.15 + 0.6, 7.0))           # радіус крони, м
        cx, cy = x * F + 1, y * F + 1
        if acc:
            if tr is None or len(acc) % 256 == 0:
                tr = cKDTree(np.array([(a[0], a[1]) for a in acc]))
            near = tr.query_ball_point((cx, cy), rm / M * 1.1)
            if any(math.hypot(acc[i][0] - cx, acc[i][1] - cy) < 0.8 * (acc[i][2] + rm / M) for i in near):
                continue
            # перевірка останніх, ще не в дереві
            if any(math.hypot(a[0] - cx, a[1] - cy) < 0.8 * (a[2] + rm / M) for a in acc[(len(acc) // 256) * 256:]):
                continue
        rp = int(round(rm / M))
        y0, y1 = max(0, y - 2), y + 3; x0, x1 = max(0, x - 2), x + 3
        c = np.median(rgb[y0:y1, x0:x1].reshape(-1, 3), 0)
        if c[1] > c[0] + 4: kind = 'leaf'
        elif c[0] > c[1] + 22: kind = 'leafO'
        else: kind = 'leafY'
        acc.append((cx, cy, rp))
        trees.append((int(cx), int(cy), rp, kind))
    return trees


def smooth_up(lab, ks, F):
    """Збільшення карти класів з ~1 м до пікселів карти з плавними (не сходинковими) краями."""
    oh = [ndi.gaussian_filter((lab == k).astype(np.float32), 0.6) for k in ks]
    out = np.zeros((H, W), np.uint8)
    xs = (np.arange(W, dtype=np.float32) + 0.5) / F - 0.5
    for y0 in range(0, H, 512):
        y1 = min(H, y0 + 512)
        ys = (np.arange(y0, y1, dtype=np.float32) + 0.5) / F - 0.5
        Y, X = np.meshgrid(ys, xs, indexing='ij')
        best = np.full(Y.shape, -1.0, np.float32); arg = np.zeros(Y.shape, np.uint8)
        for k, o in zip(ks, oh):
            v = ndi.map_coordinates(o, [Y, X], order=1, mode='nearest')
            m = v > best; best[m] = v[m]; arg[m] = k
        out[y0:y1] = arg
    return out


# ------------------------------------------------------------------ 3. будинки
ROOF_RAMPS = ['roofR', 'roofO', 'roofB', 'roofG', 'roofD', 'roofW', 'roofM', 'roofT']


def roof_choice(rgb):
    if rgb is None:
        return 'roofG', 2
    c = np.array(rgb, np.float32)
    # знімок темніший і бляклий — трохи підсилити
    c = np.clip((c - c.mean()) * 1.35 + c.mean() * 1.18 + 6, 0, 255)
    best = None
    for rp in ROOF_RAMPS:
        for lv in (1, 2, 3):
            p = np.array(PAL[R[rp][lv]], np.float32)
            dc = c - p
            dch = (c - c.mean()) - (p - p.mean())                       # відтінок важливіший за яскравість
            e = (dc ** 2).sum() * 0.5 + (dch ** 2).sum() * 1.5
            if best is None or e < best[0]: best = (e, rp, lv)
    return best[1], best[2]


def wall_px(hm):
    return int(round(np.clip(2.0 + 1.05 * hm, 4, 40)))


WALL_BY = {'house': ('wallC', 'wallY', 'wallK', 'wallW'), 'apartments': ('wallW', 'wallC', 'wallW', 'wallY'),
           'other': ('wallW', 'wallC', 'wallK', 'wallY')}


def draw_building(img, bd):
    pts = bd['pts']; h = bd['hp']
    rs = random.Random(bd['seed'])
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    bx0, by0 = int(math.floor(min(xs))) - 2, int(math.floor(min(ys))) - h - 2
    bx1, by1 = int(math.ceil(max(xs))) + 3, int(math.ceil(max(ys))) + 3
    w_, h_ = bx1 - bx0, by1 - by0
    if w_ <= 2 or h_ <= 2 or w_ > 3000 or h_ > 3000: return
    im = Image.new('1', (w_, h_), 0)
    ImageDraw.Draw(im).polygon([(x - bx0, y - by0) for x, y in pts], fill=1)
    Fm = np.array(im)
    if not Fm.any(): return
    S = Fm.copy()
    for d in range(1, h + 1): S[:-d] |= Fm[d:]
    roof = np.zeros_like(Fm); roof[:-h] = Fm[h:]
    wall = S & ~roof
    bt = bd['b']
    house = bd['pitched']
    kind = 'house' if house else ('apartments' if bt in ('apartments', 'dormitory', 'residential') else 'other')
    wramp = rs.choice(WALL_BY[kind])
    rramp, rlv = bd['roof']
    if bt in ('garage', 'garages', 'shed') and rramp not in ('roofR', 'roofO', 'roofB'):
        wramp = 'wallW'
    bay = BAYER[np.ix_(np.arange(h_) % 4, np.arange(w_) % 4)]
    yy, xx = np.mgrid[0:h_, 0:w_]
    out = np.zeros((h_, w_), np.uint8)
    base = np.full(w_, -1)
    colany = Fm.any(axis=0)
    base[colany] = (h_ - 1 - np.argmax(Fm[::-1, :], axis=0))[colany]
    hy = base[None, :] - yy
    Tw = 2.4 + np.where(hy < 2, -0.9, 0.0) + 0.4 * (hy > h - 3)
    lv = bd['lv']
    if lv >= 1 and bt not in ('garage', 'garages', 'shed', 'kiosk', 'roof') and h >= 6:
        fl = max((h - 3) / max(1, lv), 2.6)
        fpos = (hy - 2) % fl
        win = (hy >= 3) & (hy <= h - 2) & (fpos >= 0.6) & (fpos < 0.6 + min(1.6, fl * 0.5)) & ((xx % 4) < 2)
    else:
        win = np.zeros_like(wall)
    wq = MG.quant_local(Tw, wramp, bay, 0.6)
    out[wall] = wq[wall]
    gw = wall & win
    out[gw] = np.where(((xx + yy) % 5 == 0)[gw], R['glass'][2], R['glass'][1])
    rr = R[rramp]
    if house:
        d = ndi.distance_transform_edt(roof).astype(np.float32)
        Rr = max(1.5, min(d.max(), 9))
        z = np.clip(d, 0, Rr)
        gy, gx = np.gradient(z)
        Lv = np.array([-0.5, -0.65, 0.58]); Lv /= np.linalg.norm(Lv)
        n = np.sqrt(gx * gx + gy * gy + 1)
        ndl = (-gx * Lv[0] - gy * Lv[1] + Lv[2]) / n
        Tr = rlv + (ndl - Lv[2]) * 4.0
        rq = MG.quant_local(Tr, rramp, bay, 0.35)
        out[roof] = rq[roof]
    else:
        nz = np.array([[rs.random() for _ in range(w_)] for _ in range(h_)], np.float32) - 0.5
        rq = MG.quant_local(rlv + 0.5 * nz, rramp, bay, 0.7)
        out[roof] = rq[roof]
        inner = ndi.binary_erosion(roof, iterations=2)
        ring = roof & ~inner
        out[ring] = rr[min(rlv + 1, len(rr) - 1)]
        ys_, xs_ = np.nonzero(ndi.binary_erosion(roof, iterations=5))
        nv = min(len(xs_) // 160, 8)
        for _ in range(nv):
            k = rs.randrange(len(xs_)); vx, vy = xs_[k], ys_[k]
            out[vy:vy + 3, vx:vx + 4] = R['conc'][3]
            out[vy + 3, vx + 1:vx + 5] = rr[0]
            out[vy:vy + 3, vx + 4] = rr[0]
    up = np.zeros_like(roof); up[1:] = roof[:-1]
    lf = np.zeros_like(roof); lf[:, 1:] = roof[:, :-1]
    hi = roof & (~up | ~lf)
    out[hi] = rr[min(rlv + 2, len(rr) - 1)]
    rb = wall & np.vstack([roof[:1] * 0, roof[:-1]]).astype(bool)
    out[rb] = rr[0]
    ol = ndi.binary_dilation(S) & ~S
    out[ol] = R['out'][0]
    paste(img, out, bx0, by0)


def paste(img, src, x, y):
    h, w = src.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    d = img[y0:y1, x0:x1]
    m = s > 0
    d[m] = s[m]


def tree_stamp(r, seed, ramp):
    st = MG.tree_stamp(r, seed)
    if ramp == 'leaf': return st
    lut = np.arange(256, dtype=np.uint8)
    src, dst = R['leaf'], R[ramp]
    for k, i in enumerate(src): lut[i] = dst[min(k, len(dst) - 1)]
    return lut[st]


# ------------------------------------------------------------------ 4. зона гри
def play_mask():
    b = np.load('build/bounds.npz')
    play = b['play']; G = float(b['G'])
    lon0, lat1, mlon, mlat = float(b['lon0']), float(b['lat1']), float(b['mlon']), float(b['mlat'])
    xs = X0 + (np.arange(W) + 0.5) * M; ys = Y0 + (np.arange(H) + 0.5) * M
    lon = LON0 + xs / MLON; lat = LAT0 - ys / MLAT
    jx = np.clip(((lon - lon0) * mlon / G).astype(int), 0, play.shape[1] - 1)
    iy = np.clip(((lat1 - lat) * mlat / G).astype(int), 0, play.shape[0] - 1)
    return play[iy[:, None], jx[None, :]]


# ------------------------------------------------------------------ main
def main():
    nodes, ntags, ways = load('data/osm_v2')
    print('ways', len(ways), 'map', W, H)
    cls, drive, foot, rails = osm_classes(ways)

    # будинки (контури з OSM, колір/висота з buildings.json)
    BJ = json.load(open('build/buildings.json'))
    blds = []
    for k, b in enumerate(BJ['buildings']):
        t = b['t']; bt = t.get('building')
        pts = [tuple(p) for p in b['q']]
        if max(p[0] for p in pts) < -40 or min(p[0] for p in pts) > W + 40 or max(p[1] for p in pts) < -40 or min(p[1] for p in pts) > H + 80:
            continue
        hm = b['h']
        if b['h_src'] == 'shadow' and b['area'] < 150: hm = min(hm, 9.0)
        if bt in ('garage', 'garages', 'shed', 'kiosk', 'roof'): hm = min(hm, 3.5)
        lv = max(1, int(round((hm - 1.5) / 3.0)))
        pitched = (bt in ('house', 'detached', 'semidetached_house') or (b['area'] < 260 and hm < 10)) and bt not in ('garage', 'garages', 'kiosk')
        blds.append(dict(pts=pts, b=bt, area=b['area'], lv=lv, hm=hm, hp=wall_px(hm), roof=roof_choice(b['roof']),
                         pitched=pitched, seed=k * 7919 + 13, t=t))
    print('buildings', len(blds))
    sim = Image.new('1', (W, H), 0); ds = ImageDraw.Draw(sim)
    for bd in blds: ds.polygon(bd['pts'], fill=1)
    bmask = np.array(sim); del sim

    # супутникові шари
    L = satellite_layers(cls, bmask)
    F = L['F']; h1, w1 = L['h'], L['w']
    up = lambda a: np.repeat(np.repeat(a, F, 0), F, 1)[:H, :W]
    gr = smooth_up(L['ground'], (0, C_PAVED, C_ASPHY, C_DIRT, C_GRASS), F)
    over = np.isin(cls, SOFT) & (gr > 0)
    # у парках лишаємо траву, але показуємо доріжки/майданчики
    park = cls == C_GRASS
    over &= ~(park & (gr == C_DIRT))
    over &= ~((cls == C_CONC) & (gr == C_DIRT) & False)
    cls[over] = gr[over]
    del gr, over, park
    print('classes', np.bincount(cls.ravel(), minlength=17))

    # ---- текстурування смугами
    img = np.zeros((H, W), np.uint8)
    nlo = smooth_noise(26, 8); nmid = smooth_noise(4, 2)
    yel = np.zeros((H, W), np.float16); yel[:h1 * F, :w1 * F] = up(L['yel']).astype(np.float16)
    glum = np.zeros((H, W), np.float16); glum[:h1 * F, :w1 * F] = up(L['glum']).astype(np.float16)
    BS = 1024
    for y0 in range(0, H, BS):
        y1 = min(H, y0 + BS)
        c = cls[y0:y1]; o = img[y0:y1]
        yy = np.arange(y0, y1, dtype=np.int32)[:, None]; xx = np.arange(W, dtype=np.int32)[None, :]
        bay = BAYER[yy % 4, xx % 4]
        nm = nmid[y0:y1]; nl = nlo[y0:y1]
        nh = rng.rand(y1 - y0, W).astype(np.float32) - 0.5
        ye = yel[y0:y1].astype(np.float32); gl = glum[y0:y1].astype(np.float32)

        def put(cid, ramp, T, dith=0.85, m=None):
            mm = (c == cid) if m is None else m
            o[mm] = quant(T, ramp, bay, dith)[mm]
        Tg = 1.4 + 0.9 * gl + 0.35 * nm + 0.2 * nl + 0.45 * nh
        for cid in (C_YARD, C_GRASS, C_WOOD):
            mm = c == cid
            dry = mm & (ye + 0.25 * nm > 0.55)
            o[mm] = quant(Tg, 'grass', bay)[mm]
            o[dry] = quant(Tg, 'grassY', bay)[dry]
        put(C_PITCH, 'grass', 2.3 + 0.7 * ((xx // 12) % 2) + 0.2 * nm)
        put(C_TRACK, 'track', 1.8 + 0.3 * nm + 0.3 * nh, 0.6)
        put(C_DIRT, 'dirt', 2.0 + 0.5 * nm + 0.6 * nh)
        wm = c == C_WATER
        if wm.any():
            rip = np.sin(xx * 0.35 + yy * 0.08 + nm * 2.5) * 0.45
            put(C_WATER, 'water', 1.8 + rip + 0.25 * nm)
        seam6 = ((xx % 6) == 0) | ((yy % 6) == 0)
        put(C_SIDE, 'pave', 2.7 + 0.25 * nm + 0.2 * nh - 1.1 * seam6, 0.6)
        seam4 = ((xx % 5) == 0) | ((yy % 5) == 0)
        put(C_PATH, 'pave', 2.2 + 0.3 * nm + 0.3 * nh - 0.8 * seam4, 0.6)
        brick = ((yy % 4) == 0) | (((xx + ((yy // 4) % 2) * 4) % 8) == 0)
        put(C_PED, 'pave', 3.0 + 0.3 * nm + 0.15 * nl - 1.0 * brick, 0.5)
        seam12 = ((xx % 14) == 0) | ((yy % 14) == 0)
        put(C_CONC, 'conc', 2.3 + 0.3 * nm + 0.3 * nh - 0.8 * seam12)
        put(C_PAVED, 'conc', 2.6 + 0.3 * nm + 0.3 * nh - 0.7 * seam6, 0.7)
        put(C_PARKING, 'asph', 2.7 + 0.3 * nm + 0.4 * nh)
        put(C_ASPHY, 'asph', 3.1 + 0.3 * nm + 0.4 * nh, 0.7)
        put(C_ROAD, 'asph', 1.9 + 0.3 * nm + 0.2 * nl + 0.45 * nh, 0.9)
        put(C_RAIL, 'ballast', 1.7 + 1.6 * nh + 0.2 * nm, 0.7)
    del nlo, nmid, yel, glum

    # бордюри
    road = cls == C_ROAD
    walk = np.isin(cls, (C_SIDE, C_PED, C_PATH))
    curb = walk & ndi.binary_dilation(road, iterations=1)
    img[curb] = R['pave'][4]
    soft = np.isin(cls, (C_YARD, C_GRASS, C_WOOD, C_PITCH, C_DIRT))
    edge = walk & ndi.binary_dilation(soft, iterations=1) & ~curb
    img[edge] = R['pave'][1]
    del soft, edge, curb
    # розмітка паркувань (світлі смуги) на OSM-паркуваннях — пропускаємо, лише рамка
    pk = cls == C_PARKING
    img[pk & ~ndi.binary_erosion(pk, iterations=1)] = R['asph'][4]

    # рейки
    ov = Image.new('L', (W, H), 0); do = ImageDraw.Draw(ov)
    for pts in rails:
        for i in range(len(pts) - 1):
            (xa, ya), (xb, yb) = pts[i], pts[i + 1]
            Ln = math.hypot(xb - xa, yb - ya)
            if Ln < 1: continue
            ux, uy = (xb - xa) / Ln, (yb - ya) / Ln; nx, ny = -uy, ux; s = 0.0
            while s < Ln:
                cx, cy = xa + ux * s, ya + uy * s
                do.line([(cx - nx * 4.5, cy - ny * 4.5), (cx + nx * 4.5, cy + ny * 4.5)], fill=1, width=2); s += 4.5
    for pts in rails:
        for off in (-2.3, 2.3):
            sh = []
            for i in range(len(pts)):
                a = pts[max(i - 1, 0)]; b = pts[min(i + 1, len(pts) - 1)]
                Ln = math.hypot(b[0] - a[0], b[1] - a[1]) or 1
                sh.append((pts[i][0] - (b[1] - a[1]) / Ln * off, pts[i][1] + (b[0] - a[0]) / Ln * off))
            do.line(sh, fill=2, width=1)
    ova = np.array(ov); del ov
    img[ova == 1] = R['sleeper'][0]; img[ova == 2] = R['steel'][1]
    del ova

    # розмітка і переходи
    mk = Image.new('1', (W, H), 0); dm = ImageDraw.Draw(mk)
    for w in drive:
        hcl = w['t']['highway']
        if hcl not in ('primary', 'secondary', 'tertiary') or w['t'].get('oneway') == 'yes': continue
        pts = w['pts']; s0 = 0.0
        for i in range(len(pts) - 1):
            (xa, ya), (xb, yb) = pts[i], pts[i + 1]
            Ln = math.hypot(xb - xa, yb - ya); s = 0.0
            while s < Ln:
                ph = (s0 + s) % 26
                if ph < 13:
                    e = min(s + (13 - ph), Ln)
                    dm.line([(xa + (xb - xa) * s / Ln, ya + (yb - ya) * s / Ln), (xa + (xb - xa) * e / Ln, ya + (yb - ya) * e / Ln)], fill=1, width=2)
                    s = e
                else: s += 26 - ph
            s0 += Ln
    node_road = {}
    for w in drive:
        for k, nid in enumerate(w['nds']): node_road.setdefault(nid, (w, k))
    for nid, t in ntags.items():
        if t.get('highway') != 'crossing' or nid not in node_road: continue
        w, k = node_road[nid]; pts = w['pts']
        a = pts[max(k - 1, 0)]; b = pts[min(k + 1, len(pts) - 1)]
        Ln = math.hypot(b[0] - a[0], b[1] - a[1]) or 1
        ux, uy = (b[0] - a[0]) / Ln, (b[1] - a[1]) / Ln
        cx, cy = pts[k]; hw = road_width_m(w['t']) / M / 2; v = -hw
        while v < hw:
            dm.line([(cx - uy * v - ux * 7, cy + ux * v - uy * 7), (cx - uy * v + ux * 7, cy + ux * v + uy * 7)], fill=1, width=3); v += 6
    mka = np.array(mk) & road; del mk
    img[mka] = R['mark'][0]; del mka

    # ---- дерева зі знімка (+ поодинокі з OSM)
    trees = place_trees(L, bmask)
    ttree = cKDTree(np.array([(t[0], t[1]) for t in trees])) if trees else None
    for nid, t in ntags.items():
        if t.get('natural') == 'tree':
            x, y = px(*nodes[nid])
            if 0 <= x < W and 0 <= y < H and not bmask[int(y), int(x)]:
                if ttree is None or not ttree.query_ball_point((x, y), 12): trees.append((int(x), int(y), int(rng.randint(8, 13)), 'leaf'))
    print('trees', len(trees))

    # тіні (на схід від будинків, на південний схід від дерев)
    shm = Image.new('1', (W, H), 0); dsh = ImageDraw.Draw(shm)
    for bd in blds:
        sx = bd['hp'] * 0.5
        for k in range(0, int(sx) + 1, 2):
            dsh.polygon([(x + k, y - k * 0.15) for x, y in bd['pts']], fill=1)
    for (x, y, r, _) in trees:
        dsh.ellipse([x - r * 0.7 + r * 0.55, y - r * 0.45 - r * 0.2, x + r * 0.7 + r * 0.55, y + r * 0.45 - r * 0.2], fill=1)
    shm = np.array(shm) & ~bmask
    img[shm] = DARK[img[shm]]; del shm

    stamps = {}
    objs = [(max(p[1] for p in bd['pts']), 0, i) for i, bd in enumerate(blds)] + [(t[1], 1, i) for i, t in enumerate(trees)]
    objs.sort()
    for _, kind, i in objs:
        if kind == 0:
            draw_building(img, blds[i])
        else:
            x, y, r, rp = trees[i]
            r = int(np.clip(r, 6, 22))
            key = (r, i % 4, rp)
            if key not in stamps: stamps[key] = tree_stamp(r, i % 4, rp)
            st = stamps[key]; sh_, sw_ = st.shape
            paste(img, st, x - sw_ // 2, int(y - r * 0.6) - sh_ // 2)

    # ---- зона гри: поза нею — стіна
    play = play_mask()
    solid = bmask | (cls == C_WATER) | ~play
    np.save(os.path.join(OUT, 'map.npy'), img)
    np.save(os.path.join(OUT, 'solid.npy'), solid)
    np.save(os.path.join(OUT, 'play.npy'), play)

    # ---- точки забору/доставки
    access = np.isin(cls, (C_SIDE, C_PED, C_PATH, C_CONC, C_PARKING, C_PAVED, C_ASPHY)) & ~solid

    def stop_at(x, y, maxd=70):
        x, y = int(round(x)), int(round(y))
        if not (40 <= x < W - 40 and 40 <= y < H - 40): return None
        if not play[y, x] and not play[max(0, y - 60):y + 60, max(0, x - 60):x + 60].any(): return None
        xa, ya = max(0, x - maxd), max(0, y - maxd)
        sub = access[ya:y + maxd + 1, xa:x + maxd + 1]
        ys_, xs_ = np.nonzero(sub)
        if len(xs_) == 0: return None
        dd = (xs_ + xa - x) ** 2 + (ys_ + ya - y) ** 2
        k = dd.argmin()
        if dd[k] > maxd * maxd: return None
        return int(xs_[k] + xa), int(ys_[k] + ya)

    KIND = MG_KIND
    shops, seen = [], set()
    cand = [(t, px(*nodes[nid])) for nid, t in ntags.items()]
    for w in ways:
        if w['closed'] and ('shop' in w['t'] or 'amenity' in w['t']):
            cand.append((w['t'], (sum(p[0] for p in w['pts']) / len(w['pts']), sum(p[1] for p in w['pts']) / len(w['pts']))))
    for t, (x, y) in cand:
        k = t.get('shop') or t.get('amenity')
        if k not in KIND or 'name' not in t: continue
        s = stop_at(x, y)
        if not s: continue
        nm = MG.clean(t['name'])
        if not nm or (nm, s) in seen: continue
        seen.add((nm, s)); shops.append(dict(x=s[0], y=s[1], name=nm, kind=KIND[k]))
    addrs = []
    for bd in blds:
        t = bd['t']
        if 'addr:street' not in t or 'addr:housenumber' not in t: continue
        cx = sum(p[0] for p in bd['pts']) / len(bd['pts']); cy = sum(p[1] for p in bd['pts']) / len(bd['pts'])
        s = stop_at(cx, cy, 90)
        if not s: continue
        addrs.append(dict(x=s[0], y=s[1], name=MG.short_street(t['addr:street']) + ', ' + t['addr:housenumber']))
    print('shops', len(shops), 'addresses', len(addrs))

    names, segs = [], []
    for w in drive + [f for f in foot if f['t'].get('highway') == 'pedestrian']:
        nm = w['t'].get('name')
        if not nm: continue
        nm = MG.short_street(nm)
        if nm not in names: names.append(nm)
        ni = names.index(nm)
        for i in range(len(w['pts']) - 1):
            (xa, ya), (xb, yb) = w['pts'][i], w['pts'][i + 1]
            if max(xa, xb) < 0 or min(xa, xb) > W or max(ya, yb) < 0 or min(ya, yb) > H: continue
            segs.append((int(xa), int(ya), int(xb), int(yb), ni))
    spawn = None
    for w in foot + drive:
        if 'Проскурівська' in (w['t'].get('name') or ''):
            x, y = w['pts'][len(w['pts']) // 2]
            s = stop_at(x, y, 200)
            if s: spawn = s; break
    if not spawn: spawn = (W // 2, H // 2)
    json.dump(dict(shops=shops, addrs=addrs, names=names, segs=segs, spawn=spawn, W=W, H=H, M=M, X0=X0, Y0=Y0),
              open(os.path.join(OUT, 'mapdata.json'), 'w'), ensure_ascii=False)

    # ---- прев'ю
    pal = np.array(PAL + [(255, 0, 255)] * (256 - len(PAL)), np.uint8)
    small = img[::4, ::4]
    pv = pal[small].copy()
    pl = play[::4, ::4]
    pv[~pl] = (pv[~pl] * 0.55).astype(np.uint8)
    Image.fromarray(pv).save(os.path.join(OUT, 'preview_quarter.png'))
    print('done')


MG_KIND = {'cafe': "КАВ'ЯРНЯ", 'restaurant': 'РЕСТОРАН', 'fast_food': 'ФАСТФУД', 'pharmacy': 'АПТЕКА', 'supermarket': 'СУПЕРМАРКЕТ',
           'convenience': 'МАГАЗИН', 'clothes': 'ОДЯГ', 'bakery': 'ПЕКАРНЯ', 'pub': 'ПАБ', 'bar': 'БАР', 'electronics': 'ЕЛЕКТРОНІКА',
           'florist': 'КВІТИ', 'post_office': 'ПОШТА', 'chemist': 'ПОБУТХІМІЯ', 'books': 'КНИГИ', 'confectionery': 'КОНДИТЕРСЬКА',
           'computer': "КОМП'ЮТЕРИ", 'mobile_phone': 'ТЕЛЕФОНИ', 'shoes': 'ВЗУТТЯ', 'gift': 'ПОДАРУНКИ', 'cosmetics': 'КОСМЕТИКА',
           'pizza': 'ПІЦА', 'butcher': "М'ЯСО", 'alcohol': 'НАПОЇ', 'beverages': 'НАПОЇ', 'sports': 'СПОРТ', 'toys': 'ІГРАШКИ',
           'fuel': 'АЗС'}

if __name__ == '__main__':
    main()
