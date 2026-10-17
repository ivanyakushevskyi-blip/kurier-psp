"""Khmelnytskyi centre -> indexed pixel-art map (top-down, oblique buildings) + game data.
usage: python3 -I tools/mapgen.py <osm_dir> <out_dir>
"""
import math, glob, sys, os, zlib, struct, json, random
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pal import PAL, R, dark_lut

M = 0.32                    # metres per pixel
X0, Y0 = -850.0, -750.0     # world metres of pixel (0,0)
W, H = 4672, 3584
LAT0, LON0 = 49.422, 26.995
MLAT = 111320.0
MLON = 111320.0 * math.cos(math.radians(LAT0))
WIDEN = 1.2                 # roads a bit wider than life, for gameplay

C_YARD, C_GRASS, C_WOOD, C_WATER, C_SIDE, C_ROAD, C_PATH, C_PED, C_CONC, C_RAIL, C_DIRT, C_PITCH, C_PARKING = range(1, 14)

rng = np.random.RandomState(1234)
random.seed(1234)


def px(lat, lon):
    x = (lon - LON0) * MLON
    y = -(lat - LAT0) * MLAT
    return ((x - X0) / M, (y - Y0) / M)


# ------------------------------------------------------------------ load
def load(d):
    nodes, ntags, ways = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(d, 't*.osm'))):
        root = ET.parse(f).getroot()
        for n in root.iter('node'):
            nodes[n.get('id')] = (float(n.get('lat')), float(n.get('lon')))
            t = {x.get('k'): x.get('v') for x in n.iter('tag')}
            if t:
                ntags[n.get('id')] = t
        for w in root.iter('way'):
            ways[w.get('id')] = ([nd.get('ref') for nd in w.iter('nd')], {x.get('k'): x.get('v') for x in w.iter('tag')})
    out = []
    for wid, (nds, tags) in ways.items():
        ids = [i for i in nds if i in nodes]
        pts = [px(*nodes[i]) for i in ids]
        if len(pts) >= 2:
            out.append(dict(id=wid, nds=ids, pts=pts, t=tags, closed=(nds[0] == nds[-1] and len(nds) > 3)))
    return nodes, ntags, out


def poly_area(p):
    a = 0.0
    for i in range(len(p)):
        x0, y0 = p[i]; x1, y1 = p[(i + 1) % len(p)]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2


ROADW = {'primary': 14, 'secondary': 12, 'tertiary': 10, 'residential': 7, 'unclassified': 7,
         'living_street': 5.5, 'service': 4.5, 'track': 3.5, 'primary_link': 8, 'secondary_link': 7, 'tertiary_link': 7}
SIDEW = {'primary': 3.4, 'secondary': 3.2, 'tertiary': 3.0, 'residential': 2.2, 'unclassified': 2.2,
         'primary_link': 2.4, 'secondary_link': 2.2, 'tertiary_link': 2.2}


def road_width_m(t):
    h = t.get('highway')
    w = ROADW.get(h, 0)
    try:
        ln = int(t.get('lanes', '0'))
        if ln >= 2 and h in ('primary', 'secondary', 'tertiary', 'residential', 'unclassified'):
            w = max(w * 0.6, ln * 3.4)
    except ValueError:
        pass
    if t.get('service') == 'parking_aisle':
        w = 4.0
    return w * WIDEN


def draw_line(dr, pts, wpx, fill):
    if wpx < 1:
        wpx = 1
    dr.line(pts, fill=fill, width=int(round(wpx)))
    r = wpx / 2.0
    for (x, y) in pts:
        dr.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def smooth_noise(sigma, down):
    h, w = H // down + 2, W // down + 2
    n = ndi.gaussian_filter(rng.randn(h, w).astype(np.float32), sigma / down)
    n /= (n.std() + 1e-6)
    n = ndi.zoom(n, down, order=1)[:H, :W]
    return n.astype(np.float32)


def quant(T, ramp, bay, dith=0.85):
    ids = np.array(R[ramp], np.uint8)
    i = np.floor(T + (bay - 0.5) * dith + 0.5)
    i = np.clip(i, 0, len(ids) - 1).astype(np.int32)
    return ids[i]


# ------------------------------------------------------------------ main
def main(osm_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    nodes, ntags, ways = load(osm_dir)
    print('ways', len(ways))

    cls_im = Image.new('L', (W, H), C_YARD)
    dc = ImageDraw.Draw(cls_im)

    # ---- areas
    areas = []
    for w in ways:
        if not w['closed']:
            continue
        t = w['t']
        c = None
        lu, le, na, am = t.get('landuse'), t.get('leisure'), t.get('natural'), t.get('amenity')
        if lu in ('retail', 'commercial', 'industrial', 'garages'): c = C_CONC
        elif lu in ('construction', 'brownfield'): c = C_DIRT
        elif lu in ('grass', 'meadow', 'recreation_ground', 'village_green', 'flowerbed', 'cemetery'): c = C_GRASS
        elif le in ('park', 'garden', 'playground', 'dog_park'): c = C_GRASS
        elif le in ('pitch', 'track'): c = C_PITCH
        elif le in ('stadium', 'sports_centre'): c = C_GRASS
        elif na in ('wood', 'scrub') or lu == 'forest': c = C_WOOD
        elif na in ('grassland', 'heath'): c = C_GRASS
        elif na == 'water' or lu in ('reservoir', 'basin') or le == 'swimming_pool': c = C_WATER
        elif am == 'parking': c = C_PARKING
        elif t.get('highway') == 'pedestrian' and t.get('area') == 'yes': c = C_PED
        elif t.get('place') == 'square' or (t.get('area:highway') == 'pedestrian'): c = C_PED
        elif t.get('railway') == 'platform': c = C_CONC
        if c is not None:
            areas.append((poly_area(w['pts']), c, w['pts']))
    areas.sort(key=lambda a: -a[0])
    for _, c, pts in areas:
        dc.polygon(pts, fill=c)

    # ---- water lines
    WW = {'river': 22, 'canal': 10, 'stream': 4, 'ditch': 2, 'drain': 2}
    for w in ways:
        ww = w['t'].get('waterway')
        if ww in WW and w['t'].get('tunnel') not in ('yes', 'culvert'):
            draw_line(dc, w['pts'], WW[ww] / M, C_WATER)

    # ---- rail ballast
    rails = []
    for w in ways:
        rw = w['t'].get('railway')
        if rw in ('rail', 'narrow_gauge', 'disused', 'siding') and w['t'].get('tunnel') != 'yes':
            draw_line(dc, w['pts'], 4.2 / M, C_RAIL)
            rails.append(w['pts'])

    # ---- roads
    drive, foot = [], []
    for w in ways:
        t = w['t']; h = t.get('highway')
        if not h or t.get('tunnel') == 'yes' or t.get('area') == 'yes':
            continue
        if h in ROADW:
            drive.append(w)
        elif h in ('footway', 'path', 'cycleway', 'steps', 'pedestrian', 'bridleway', 'corridor'):
            foot.append(w)
    # sidewalks (wide, under the carriageway)
    for w in drive:
        h = w['t']['highway']
        if h in SIDEW:
            draw_line(dc, w['pts'], (road_width_m(w['t']) + 2 * SIDEW[h] * WIDEN) / M, C_SIDE)
    for w in foot:
        h = w['t']['highway']
        if h == 'pedestrian':
            draw_line(dc, w['pts'], 11 / M, C_PED)
        elif w['t'].get('footway') in ('sidewalk', 'crossing'):
            draw_line(dc, w['pts'], 3.0 / M, C_SIDE)
        else:
            draw_line(dc, w['pts'], 2.4 / M, C_PATH)
    for w in drive:
        draw_line(dc, w['pts'], road_width_m(w['t']) / M, C_ROAD)

    cls = np.array(cls_im)
    print('classes', np.bincount(cls.ravel(), minlength=14))

    # ---------------------------------------------------------- texturing
    yy, xx = np.mgrid[0:H, 0:W]
    BAY = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0
    bay = BAY[yy % 4, xx % 4]
    nlo = smooth_noise(26, 8)
    nmid = smooth_noise(4, 2)
    nhi = rng.rand(H, W).astype(np.float32) - 0.5
    img = np.zeros((H, W), np.uint8)

    def put(cid, ramp, T, dith=0.85):
        m = cls == cid
        img[m] = quant(T, ramp, bay, dith)[m]

    # yard: patchy grass with trodden dirt
    Tg = 2.2 + 0.5 * nmid + 0.35 * nlo + 0.5 * nhi
    put(C_YARD, 'grass', Tg)
    dirtm = (cls == C_YARD) & (nlo + 0.6 * nmid < -1.05)
    img[dirtm] = quant(2.1 + 0.5 * nmid + 0.4 * nhi, 'dirt', bay)[dirtm]
    put(C_GRASS, 'grass', 2.6 + 0.45 * nmid + 0.3 * nlo + 0.5 * nhi)
    put(C_WOOD, 'grass', 1.5 + 0.45 * nmid + 0.4 * nhi)
    put(C_PITCH, 'grass', 2.3 + 0.7 * ((xx // 12) % 2) + 0.2 * nmid)
    put(C_DIRT, 'dirt', 2.0 + 0.55 * nmid + 0.6 * nhi)
    # water with ripples, brighter near the bank
    wm = cls == C_WATER
    dbank = ndi.distance_transform_edt(wm).astype(np.float32)
    rip = np.sin(xx * 0.35 + yy * 0.08 + nmid * 2.5) * 0.45
    img[wm] = quant(1.8 + rip + 0.25 * nmid + np.clip(2.5 - dbank, 0, 2.5) * 0.45, 'water', bay)[wm]
    # pavements
    seam6 = ((xx % 6) == 0) | ((yy % 6) == 0)
    put(C_SIDE, 'pave', 2.7 + 0.25 * nmid + 0.2 * nhi - 1.1 * seam6, 0.6)
    seam4 = ((xx % 5) == 0) | ((yy % 5) == 0)
    put(C_PATH, 'pave', 2.2 + 0.3 * nmid + 0.3 * nhi - 0.8 * seam4, 0.6)
    brick = ((yy % 4) == 0) | (((xx + ((yy // 4) % 2) * 4) % 8) == 0)
    put(C_PED, 'pave', 3.0 + 0.3 * nmid + 0.15 * nlo - 1.0 * brick, 0.5)
    seam12 = ((xx % 14) == 0) | ((yy % 14) == 0)
    put(C_CONC, 'conc', 2.3 + 0.3 * nmid + 0.3 * nhi - 0.8 * seam12)
    put(C_PARKING, 'asph', 2.7 + 0.3 * nmid + 0.4 * nhi)
    patch = (nlo > 1.5) | (smooth_noise(6, 2) > 2.0)
    put(C_ROAD, 'asph', 1.9 + 0.3 * nmid + 0.2 * nlo + 0.45 * nhi - 0.7 * patch, 0.9)
    put(C_RAIL, 'ballast', 1.7 + 1.6 * nhi + 0.2 * nmid, 0.7)
    del nhi

    # curbs and edges
    road = cls == C_ROAD
    walk = np.isin(cls, (C_SIDE, C_PED, C_PATH))
    curb = walk & ndi.binary_dilation(road, iterations=1)
    img[curb] = R['pave'][4]
    soft = np.isin(cls, (C_YARD, C_GRASS, C_WOOD, C_PITCH, C_DIRT))
    edge = walk & ndi.binary_dilation(soft, iterations=1) & ~curb
    img[edge] = R['pave'][1]

    # rails: sleepers + steel
    ov = Image.new('L', (W, H), 0); do = ImageDraw.Draw(ov)
    for pts in rails:
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 1: continue
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            nx, ny = -uy, ux
            s = 0.0
            while s < L:
                cx, cy = x0 + ux * s, y0 + uy * s
                do.line([(cx - nx * 4.5, cy - ny * 4.5), (cx + nx * 4.5, cy + ny * 4.5)], fill=1, width=2)
                s += 4.5
    for pts in rails:
        for off in (-2.3, 2.3):
            sh = []
            for i in range(len(pts)):
                a = pts[max(i - 1, 0)]; b = pts[min(i + 1, len(pts) - 1)]
                L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1
                sh.append((pts[i][0] - (b[1] - a[1]) / L * off, pts[i][1] + (b[0] - a[0]) / L * off))
            do.line(sh, fill=2, width=1)
    ova = np.array(ov)
    img[ova == 1] = R['sleeper'][0]
    img[ova == 2] = R['steel'][1]

    # lane markings + crossings
    mk = Image.new('L', (W, H), 0); dm = ImageDraw.Draw(mk)
    for w in drive:
        h = w['t']['highway']
        if h not in ('primary', 'secondary', 'tertiary') or w['t'].get('oneway') == 'yes':
            continue
        pts = w['pts']
        if h == 'primary' and road_width_m(w['t']) > 14:
            for off in (-1.6, 1.6):
                sh = []
                for i in range(len(pts)):
                    a = pts[max(i - 1, 0)]; b = pts[min(i + 1, len(pts) - 1)]
                    L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1
                    sh.append((pts[i][0] - (b[1] - a[1]) / L * off, pts[i][1] + (b[0] - a[0]) / L * off))
                dm.line(sh, fill=1, width=1)
            continue
        s0 = 0.0
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            L = math.hypot(x1 - x0, y1 - y0)
            s = 0.0
            while s < L:
                ph = (s0 + s) % 26
                if ph < 13:
                    e = min(s + (13 - ph), L)
                    dm.line([(x0 + (x1 - x0) * s / L, y0 + (y1 - y0) * s / L), (x0 + (x1 - x0) * e / L, y0 + (y1 - y0) * e / L)], fill=1, width=2)
                    s = e
                else:
                    s += 26 - ph
            s0 += L
    # zebra crossings
    node_road = {}
    for w in drive:
        for k, nid in enumerate(w['nds']):
            node_road.setdefault(nid, (w, k))
    ncross = 0
    for nid, t in ntags.items():
        if t.get('highway') != 'crossing' or nid not in node_road:
            continue
        w, k = node_road[nid]
        pts = w['pts']
        a = pts[max(k - 1, 0)]; b = pts[min(k + 1, len(pts) - 1)]
        L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        cx, cy = pts[k]
        hw = road_width_m(w['t']) / M / 2
        v = -hw
        while v < hw:
            p0 = (cx - uy * v - ux * 7, cy + ux * v - uy * 7)
            p1 = (cx - uy * v + ux * 7, cy + ux * v + uy * 7)
            dm.line([p0, p1], fill=1, width=3)
            v += 6
        ncross += 1
    mka = (np.array(mk) == 1) & road
    img[mka] = R['mark'][0]
    print('crossings', ncross)

    # ---------------------------------------------------------- buildings & trees
    blds = []
    for w in ways:
        t = w['t']
        if 'building' not in t or not w['closed']:
            continue
        pts = w['pts']
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if max(xs) < -40 or min(xs) > W + 40 or max(ys) < -40 or min(ys) > H + 60:
            continue
        area_m2 = poly_area(pts) * M * M
        b = t.get('building')
        try:
            lv = int(float(t.get('building:levels', '0')))
        except ValueError:
            lv = 0
        if lv <= 0:
            if b in ('house', 'detached', 'semidetached_house', 'garage', 'garages', 'shed', 'hut', 'kiosk', 'roof'):
                lv = 1
            elif b in ('apartments', 'dormitory'):
                lv = 5
            elif area_m2 < 120: lv = 1
            elif area_m2 < 400: lv = 2
            else: lv = 4
        if b in ('garage', 'garages', 'shed', 'kiosk', 'roof'):
            lv = 1
        blds.append(dict(pts=pts, lv=lv, b=b, area=area_m2, id=w['id'], t=t))
    print('buildings', len(blds))

    solid = Image.new('1', (W, H), 0); ds = ImageDraw.Draw(solid)
    for bd in blds:
        ds.polygon(bd['pts'], fill=1)
    solid = np.array(solid)
    solid |= (cls == C_WATER)

    # tree candidates
    trees = []
    occ = set()
    bdil = ndi.binary_dilation(np.array(solid) & (cls != C_WATER), iterations=5)
    plantable = np.isin(cls, (C_YARD, C_GRASS, C_WOOD)) & ~bdil
    near_walk = ndi.binary_dilation(walk | road, iterations=3)
    dens = {C_YARD: 1 / 1500.0, C_GRASS: 1 / 950.0, C_WOOD: 1 / 190.0}
    for cid, d in dens.items():
        ys_, xs_ = np.nonzero((cls == cid) & plantable)
        n = int(len(xs_) * d)
        if n == 0: continue
        sel = rng.choice(len(xs_), size=n, replace=False)
        for i in sel:
            x, y = int(xs_[i]), int(ys_[i])
            if near_walk[y, x] and cid != C_WOOD:
                continue
            cell = (x // 12, y // 12) if cid == C_WOOD else (x // 18, y // 18)
            if cell in occ: continue
            occ.add(cell)
            r = rng.randint(9, 14) if cid == C_WOOD else rng.randint(7, 13)
            trees.append((x, y, r))
    for w in ways:
        if w['t'].get('natural') == 'tree_row':
            pts = w['pts']
            for i in range(len(pts) - 1):
                (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                L = math.hypot(x1 - x0, y1 - y0); s = 6.0
                while s < L:
                    x, y = int(x0 + (x1 - x0) * s / L), int(y0 + (y1 - y0) * s / L)
                    if 0 <= x < W and 0 <= y < H and not solid[y, x]:
                        trees.append((x, y, rng.randint(7, 10)))
                    s += 17
    for nid, t in ntags.items():
        if t.get('natural') == 'tree':
            x, y = px(*nodes[nid])
            if 0 <= x < W and 0 <= y < H and not solid[int(y), int(x)]:
                trees.append((int(x), int(y), rng.randint(7, 11)))
    print('trees', len(trees))

    DARK = np.array(dark_lut(), np.uint8)

    # shadows on the ground (east side of buildings, south-east of trees)
    shm = Image.new('1', (W, H), 0); dsh = ImageDraw.Draw(shm)
    for bd in blds:
        h = bheight(bd)
        sx = h * 0.55
        for k in range(0, int(sx) + 1, 2):
            dsh.polygon([(x + k, y - k * 0.15) for x, y in bd['pts']], fill=1)
    for (x, y, r) in trees:
        dsh.ellipse([x - r * 0.7 + r * 0.55, y - r * 0.45 - r * 0.2, x + r * 0.7 + r * 0.55, y + r * 0.45 - r * 0.2], fill=1)
    shm = np.array(shm) & ~solid
    sm = shm & (((xx + yy) % 2 == 0) | True)
    img[sm] = DARK[img[sm]]
    del xx, yy, bay

    # tree stamps
    stamps = {}
    for r in range(7, 15):
        stamps[r] = [tree_stamp(r, s) for s in range(4)]

    objs = [(max(p[1] for p in bd['pts']), 0, i) for i, bd in enumerate(blds)] + [(t[1], 1, i) for i, t in enumerate(trees)]
    objs.sort()
    for _, kind, i in objs:
        if kind == 0:
            draw_building(img, blds[i])
        else:
            x, y, r = trees[i]
            st = stamps[r][i % 4]
            sh_, sw_ = st.shape
            paste(img, st, x - sw_ // 2, int(y - r * 0.6) - sh_ // 2)

    # ---------------------------------------------------------- outputs
    Image.fromarray(img, 'P').save(os.path.join(out_dir, 'map_full.png'), optimize=False) if False else None
    np.save(os.path.join(out_dir, 'map.npy'), img)
    np.save(os.path.join(out_dir, 'solid.npy'), solid)
    np.save(os.path.join(out_dir, 'cls.npy'), cls)

    # stops for pickups / drop-offs
    access = np.isin(cls, (C_SIDE, C_PED, C_PATH, C_CONC, C_PARKING)) & ~solid
    dist, (iy, ix) = ndi.distance_transform_edt(~access, return_indices=True)

    def stop_at(x, y, maxd=70):
        x, y = int(round(x)), int(round(y))
        if not (40 <= x < W - 40 and 40 <= y < H - 40):
            return None
        if dist[y, x] > maxd:
            return None
        return int(ix[y, x]), int(iy[y, x])

    KIND = {'cafe': 'КАВ\'ЯРНЯ', 'restaurant': 'РЕСТОРАН', 'fast_food': 'ФАСТФУД', 'pharmacy': 'АПТЕКА', 'supermarket': 'СУПЕРМАРКЕТ',
            'convenience': 'МАГАЗИН', 'clothes': 'ОДЯГ', 'bakery': 'ПЕКАРНЯ', 'pub': 'ПАБ', 'bar': 'БАР', 'electronics': 'ЕЛЕКТРОНІКА',
            'florist': 'КВІТИ', 'post_office': 'ПОШТА', 'chemist': 'ПОБУТХІМІЯ', 'books': 'КНИГИ', 'confectionery': 'КОНДИТЕРСЬКА',
            'computer': 'КОМП\'ЮТЕРИ', 'mobile_phone': 'ТЕЛЕФОНИ', 'shoes': 'ВЗУТТЯ', 'gift': 'ПОДАРУНКИ', 'cosmetics': 'КОСМЕТИКА',
            'pizza': 'ПІЦА', 'butcher': 'М\'ЯСО', 'alcohol': 'НАПОЇ', 'beverages': 'НАПОЇ', 'sports': 'СПОРТ', 'toys': 'ІГРАШКИ'}
    shops = []
    seen = set()
    for nid, t in ntags.items():
        k = t.get('shop') or t.get('amenity')
        if k not in KIND or 'name' not in t:
            continue
        x, y = px(*nodes[nid])
        s = stop_at(x, y)
        if not s: continue
        nm = clean(t['name'])
        if not nm or (nm, s) in seen: continue
        seen.add((nm, s))
        shops.append(dict(x=s[0], y=s[1], name=nm, kind=KIND[k]))
    addrs = []
    for bd in blds:
        t = bd['t']
        if 'addr:street' not in t or 'addr:housenumber' not in t:
            continue
        cx = sum(p[0] for p in bd['pts']) / len(bd['pts']); cy = sum(p[1] for p in bd['pts']) / len(bd['pts'])
        s = stop_at(cx, cy, 90)
        if not s: continue
        addrs.append(dict(x=s[0], y=s[1], name=short_street(t['addr:street']) + ', ' + t['addr:housenumber']))
    print('shops', len(shops), 'addresses', len(addrs))

    # street-name segments
    names, segs = [], []
    for w in drive + [f for f in foot if f['t'].get('highway') == 'pedestrian']:
        nm = w['t'].get('name')
        if not nm: continue
        nm = short_street(nm)
        if nm not in names: names.append(nm)
        ni = names.index(nm)
        for i in range(len(w['pts']) - 1):
            (x0, y0), (x1, y1) = w['pts'][i], w['pts'][i + 1]
            if max(x0, x1) < 0 or min(x0, x1) > W or max(y0, y1) < 0 or min(y0, y1) > H: continue
            segs.append((int(x0), int(y0), int(x1), int(y1), ni))
    print('street names', len(names), 'segments', len(segs))

    # spawn point: on Proskurivska (pedestrian centre) if present
    spawn = None
    for w in foot + drive:
        if 'Проскурівська' in (w['t'].get('name') or ''):
            x, y = w['pts'][len(w['pts']) // 2]
            s = stop_at(x, y, 200)
            if s: spawn = s; break
    if not spawn: spawn = (W // 2, H // 2)

    json.dump(dict(shops=shops, addrs=addrs, names=names, segs=segs, spawn=spawn, W=W, H=H, M=M),
              open(os.path.join(out_dir, 'mapdata.json'), 'w'), ensure_ascii=False)
    print('done')


def clean(s):
    s = s.replace('«', '"').replace('»', '"').replace('“', '"').replace('”', '"').replace('’', "'").replace('ʼ', "'")
    s = s.strip()
    if len(s) > 22:
        s = s[:21].rstrip() + '.'
    return s


def short_street(n):
    n = n.replace('’', "'").replace('ʼ', "'")
    for full, ab in (('вулиця', 'вул.'), ('провулок', 'пров.'), ('проспект', 'просп.'), ('площа', 'пл.'),
                     ('проїзд', 'пр-д'), ('шосе', 'шосе'), ('бульвар', 'бульв.'), ('майдан', 'майдан')):
        if full in n.split():
            rest = ' '.join(x for x in n.split() if x != full)
            return (ab + ' ' + rest) if ab not in ('шосе',) else (rest + ' ' + ab)
    return n


def bheight(bd):
    return 5 + 3.0 * min(bd['lv'], 7)


WALL_BY = {'house': ('wallC', 'wallY', 'wallK', 'wallW'), 'apartments': ('wallW', 'wallC', 'wallW', 'wallY'),
           'other': ('wallW', 'wallC', 'wallK', 'wallY')}


def draw_building(img, bd):
    pts = bd['pts']
    h = int(round(bheight(bd)))
    seed = int(bd['id']) % 100000
    rs = random.Random(seed)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    bx0, by0 = int(math.floor(min(xs))) - 2, int(math.floor(min(ys))) - h - 2
    bx1, by1 = int(math.ceil(max(xs))) + 3, int(math.ceil(max(ys))) + 3
    w_, h_ = bx1 - bx0, by1 - by0
    if w_ <= 2 or h_ <= 2 or w_ > 2000 or h_ > 2000:
        return
    im = Image.new('1', (w_, h_), 0)
    ImageDraw.Draw(im).polygon([(x - bx0, y - by0) for x, y in pts], fill=1)
    F = np.array(im)
    if not F.any():
        return
    S = F.copy()
    for d in range(1, h + 1):
        S[:-d] |= F[d:]
    roof = np.zeros_like(F); roof[:-h] = F[h:]
    wall = S & ~roof
    small = bd['area'] < 260 and bd['lv'] <= 2
    house = bd['b'] in ('house', 'detached', 'semidetached_house') or small
    kind = 'house' if house else ('apartments' if bd['b'] in ('apartments', 'dormitory', 'residential') else 'other')
    wramp = rs.choice(WALL_BY[kind])
    if house:
        rramp = rs.choices(['roofR', 'roofB', 'roofM', 'roofG'], [5, 3, 2, 1])[0]
    else:
        rramp = rs.choices(['roofG', 'roofD', 'roofB', 'roofR'], [6, 4, 1, 1])[0]
    if bd['b'] in ('garage', 'garages', 'shed'):
        rramp, wramp = 'roofD', 'wallW'
    bay = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32)[np.ix_(np.arange(h_) % 4, np.arange(w_) % 4)] / 16
    yy, xx = np.mgrid[0:h_, 0:w_]
    out = np.zeros((h_, w_), np.uint8)
    # wall: height above ground per column
    base = np.full(w_, -1)
    colany = F.any(axis=0)
    base[colany] = (h_ - 1 - np.argmax(F[::-1, :], axis=0))[colany]
    hy = base[None, :] - yy
    Tw = 2.4 + np.where(hy < 2, -0.9, 0.0) + 0.4 * (hy > h - 3)
    # windows
    if bd['lv'] >= 1 and bd['b'] not in ('garage', 'garages', 'shed', 'kiosk'):
        fl = (h - 4) / max(1, min(bd['lv'], 7))
        fl = max(fl, 2.6)
        fpos = (hy - 2) % fl
        win = (hy >= 3) & (hy <= h - 2) & (fpos >= 0.6) & (fpos < 2.0) & ((xx % 4) < 2)
    else:
        win = np.zeros_like(wall)
    wq = quant_local(Tw, wramp, bay, 0.6)
    out[wall] = wq[wall]
    gw = wall & win
    out[gw] = np.where(((xx + yy) % 5 == 0)[gw], R['glass'][2], R['glass'][1])
    # door strip at ground on bigger buildings
    # roof
    if house and rramp != 'roofG':
        d = ndi.distance_transform_edt(roof).astype(np.float32)
        Rr = max(1.5, min(d.max(), 9))
        z = np.clip(d, 0, Rr)
        gy, gx = np.gradient(z)
        L = np.array([-0.5, -0.65, 0.58]); L /= np.linalg.norm(L)
        n = np.sqrt(gx * gx + gy * gy + 1)
        ndl = (-gx * L[0] - gy * L[1] + L[2]) / n
        Tr = 2.2 + (ndl - L[2]) * 4.5
        rq = quant_local(Tr, rramp, bay, 0.35)
        out[roof] = rq[roof]
    else:
        nz = np.array([[rs.random() for _ in range(w_)] for _ in range(h_)], np.float32) - 0.5
        Tr = 2.0 + 0.6 * nz
        rq = quant_local(Tr, rramp, bay, 0.8)
        out[roof] = rq[roof]
        inner = ndi.binary_erosion(roof, iterations=2)
        ring = roof & ~inner
        rr = R[rramp]
        out[ring] = rr[min(3, len(rr) - 1)]
        # vents / AC boxes
        ys_, xs_ = np.nonzero(ndi.binary_erosion(roof, iterations=5))
        nv = min(len(xs_) // 140, 6)
        for _ in range(nv):
            k = rs.randrange(len(xs_)); vx, vy = xs_[k], ys_[k]
            out[vy:vy + 3, vx:vx + 4] = R['conc'][3]
            out[vy + 3, vx + 1:vx + 5] = rr[0]
            out[vy:vy + 3, vx + 4] = rr[0]
    # edge light/dark on roof
    up = np.zeros_like(roof); up[1:] = roof[:-1]
    lf = np.zeros_like(roof); lf[:, 1:] = roof[:, :-1]
    hi = roof & (~up | ~lf)
    out[hi] = R[rramp][min(4, len(R[rramp]) - 1)]
    # roof/wall boundary line
    rb = wall & np.vstack([roof[:1] * 0, roof[:-1]]).astype(bool)
    out[rb] = R[rramp][0]
    # outline
    U = S
    ol = ndi.binary_dilation(U) & ~U
    out[ol] = R['out'][0]
    m = out > 0
    paste(img, out, bx0, by0)


def quant_local(T, ramp, bay, dith):
    ids = np.array(R[ramp], np.uint8)
    i = np.clip(np.floor(T + (bay - 0.5) * dith + 0.5), 0, len(ids) - 1).astype(np.int32)
    return ids[i]


def tree_stamp(r, seed):
    rs = np.random.RandomState(seed * 31 + r)
    S = 2 * r + 6
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    c = S / 2
    m = np.zeros((S, S), bool)
    m |= (xx - c) ** 2 + (yy - c) ** 2 < (r * 0.78) ** 2
    for k in range(9):
        a = rs.rand() * 2 * math.pi
        d = r * (0.35 + rs.rand() * 0.35)
        rr = r * (0.32 + rs.rand() * 0.22)
        m |= (xx - c - math.cos(a) * d) ** 2 + (yy - c - math.sin(a) * d) ** 2 < rr ** 2
    d = ndi.distance_transform_edt(m).astype(np.float32)
    z = np.sqrt(np.clip(d / max(r * 0.6, 1), 0, 1)) * r * 0.8
    z += ndi.gaussian_filter(rs.randn(S, S).astype(np.float32), 1.0) * 1.3
    gy, gx = np.gradient(z)
    L = np.array([-0.5, -0.7, 0.5]); L /= np.linalg.norm(L)
    n = np.sqrt(gx * gx + gy * gy + 1)
    ndl = (-gx * L[0] - gy * L[1] + L[2]) / n
    T = 2.6 + (ndl - L[2]) * 3.2 - (yy - c) / r * 0.6
    bay = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32)[np.ix_(np.arange(S) % 4, np.arange(S) % 4)] / 16
    q = quant_local(T, 'leaf', bay, 0.9)
    out = np.where(m, q, 0).astype(np.uint8)
    edge = m & ~ndi.binary_erosion(m)
    out[edge] = R['leaf'][0]
    spark = m & (rs.rand(S, S) < 0.05) & (ndl > L[2] + 0.1)
    out[spark] = R['leaf'][5]
    ol = ndi.binary_dilation(m) & ~m
    out[ol & (yy > c)] = R['out'][0]
    return out


def paste(img, src, x, y):
    h, w = src.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    d = img[y0:y1, x0:x1]
    m = s > 0
    d[m] = s[m]


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
