"""Уся карта, намальована citydraw, у масштабі гри 0,5 м/пікс.
Вихід: build/map.npy (індекси палітри palgame), build/solid.npy, build/play.npy, build/mapdata.json,
       build/map_colors.json, build/preview_full.jpg
Велика карта (v2, ≈13 400×10 300): блоки поза зоною з запасом не малюються; палітра — через 24-бітну таблицю
(без np.unique по всій карті); точки замовлень/адрес/назв вулиць — прямо з OSM у сітці гри.
Паралельно: python3 -I tools/citybuild.py [процесів=2]
"""
import sys, os, json, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

import citydraw as CD

META = json.load(open('build/sat_map.json'))
K = CD.K
CS = 64
W = int(META['W'] * K) // CS * CS
H = int(META['H'] * K) // CS * CS
SS = int(os.environ.get('SS', '4'))              # надвибірка (згладжені краї); 1 — стара піксельна карта
B, M = (1024 if SS == 1 else 512), 96
FS = 0.5 / CD.MPP                         # масштаб відносно старої сітки 0,5 м/пікс (1,8 з 31.10.2026)
from numpy.lib.format import open_memmap
LAT0, LON0 = 49.422, 26.995
MLAT = 111320.0; MLON = 111320.0 * math.cos(math.radians(LAT0))
G_YARD, G_GRASS, G_WATER, G_SIDE, G_ROAD, G_PATH, G_PED, G_RAIL, G_DIRT, G_PITCH, G_TRACK, G_PARK, G_PAVED, G_WOOD, G_SAND = range(1, 16)


_ZB = {}
def zone(key, r0=0, r1=None, step=1):
    """play / margin з build/bounds.npz у сітці гри: рядки [r0, r1) з кроком step (кроком — для грубих оцінок)."""
    if key not in _ZB:
        b = np.load('build/bounds.npz'); _ZB[key] = (b[key], float(b['G']), float(b['lon0']), float(b['lat1']), float(b['mlon']), float(b['mlat']))
    z, G, lon0, lat1, mlon, mlat = _ZB[key]
    r1 = H if r1 is None else r1
    xs = META['x0'] + (np.arange(0, W, step) + 0.5) * CD.MPP; ys = META['y0'] + (np.arange(r0, r1, step) + 0.5) * CD.MPP
    lon = LON0 + xs / MLON; lat = LAT0 - ys / MLAT
    jx = np.clip(((lon - lon0) * mlon / G).astype(int), 0, z.shape[1] - 1)
    iy = np.clip(((lat1 - lat) * mlat / G).astype(int), 0, z.shape[0] - 1)
    return z[iy[:, None], jx[None, :]]


def arch_poly(Q, nmax=64):
    """контур будинку над аркою — не більше nmax вершин (раніше обрізався до 24 → дах трикутником)"""
    eps = 1.0
    while True:
        q = cv2.approxPolyDP(np.asarray(Q, np.float32), eps, True)[:, 0, :]
        if len(q) <= nmax: return [[int(round(x)), int(round(y))] for x, y in q]
        eps *= 1.4


BLK = 'build/blocks_ss%d' % SS            # готові блоки (продовження після перерви)


def render_block(args):
    bx, by = args
    w, h = min(B, W - bx), min(B, H - by)
    fn = os.path.join(BLK, '%d_%d.npz' % (bx, by))
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=True)
        return bx, by, z['img'], z['cls'], [tuple(t) for t in z['trees'].tolist()], list(z['solid'].tolist())
    tl, so = [], []
    img, c = CD.render(bx - M, by - M, w + 2 * M, h + 2 * M, tree_out=tl, ss=SS, solid_out=so)
    trees = [t for t in tl if bx <= t[0] < bx + w and by <= t[1] < by + h]
    img, c = img[M:M + h, M:M + w].copy(), c[M:M + h, M:M + w].copy()
    so_ = np.empty(len(so), dtype=object); so_[:] = so
    tr_ = np.empty(len(trees), dtype=object); tr_[:] = trees
    np.savez(fn + '.tmp.npz', img=img, cls=c, trees=tr_, solid=so_)
    os.replace(fn + '.tmp.npz', fn)
    return bx, by, img, c, trees, so


EXTRA_SOLID = []                             # паркани, сараї, теплиці (світові координати сітки гри)


def render_all(jobs, rgb, cls, nproc):
    os.makedirs(BLK, exist_ok=True)
    trees = []
    CD.data()                                    # кеш OSM/будинків до розгалуження процесів
    if nproc > 1:
        import multiprocessing as mp
        pool = mp.get_context('fork').Pool(nproc, maxtasksperchild=8)
        it = pool.imap_unordered(render_block, jobs)
    else:
        it = map(render_block, jobs)
    for k, (bx, by, img, c, tl, so) in enumerate(it):
        h, w = c.shape
        rgb[by:by + h, bx:bx + w] = img; cls[by:by + h, bx:bx + w] = c; trees.extend(tl); EXTRA_SOLID.extend(so)
        print('  block %d/%d' % (k + 1, len(jobs)), bx, by, flush=True)
    if nproc > 1: pool.close(); pool.join()
    # ---- дерева одним проходом (без швів між блоками)
    for x, y, rr, kind in sorted(trees, key=lambda t: (t[1], t[0])):
        if SS > 1: CD.draw_tree_aa(rgb, x, y, rr, kind, random.Random(x * 7919 + y), SS)
        else: CD.draw_tree(rgb, x, y, rr, kind, random.Random(x * 7919 + y))
    print('trees', len(trees), flush=True)
    json.dump([[int(x), int(y), round(float(rr), 1), k] for x, y, rr, k in trees], open('build/trees.json', 'w'))


ARCH_ROOFS = []
BAR_RED, BAR_WHITE = (200, 56, 44), (236, 236, 230)


def barriers(rgb, cls, play):
    """Межа зони на дорогах: червоно-білий бетонний блок поперек проїзду (смуга ≈2,5 м уздовж межі).
    Лише там, де дорога перетинає межу; де дорога йде вздовж межі — нічого (довгі смуги відкидаються). Смугами рядків."""
    R = 1024; PAD = int(60 * FS); n_ok = 0
    k5 = max(3, int(round(5 * FS)) | 1); w6 = 6 * FS; w5 = 5 * FS
    for r0 in range(0, H, R):
        a0, a1 = max(0, r0 - PAD), min(H, r0 + R + PAD)
        pl = np.asarray(play[a0:a1]).astype(np.uint8)
        dist = cv2.distanceTransform(pl, cv2.DIST_L2, 3)
        road = cv2.dilate((np.asarray(cls[a0:a1]) == G_ROAD).astype(np.uint8), np.ones((k5, k5), np.uint8))
        band = (road > 0) & (dist > 0) & (dist <= w6)
        n, lab, st, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
        ok = np.zeros(n, bool)
        ok[1:] = (st[1:, cv2.CC_STAT_AREA] >= 20 * FS * FS) & (st[1:, cv2.CC_STAT_AREA] <= 420 * FS * FS) & \
                 (np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) <= 70 * FS)
        kp = ok[lab][r0 - a0:r0 - a0 + min(R, H - r0)]; d = dist[r0 - a0:r0 - a0 + kp.shape[0]]
        yy, xx = np.mgrid[r0:r0 + d.shape[0], 0:W]
        red = ((xx + yy) // int(6 * FS)) % 2 == 0
        blk = np.asarray(rgb[r0:r0 + d.shape[0]])
        inner = kp & (d > FS) & (d <= w5)
        blk[inner & red] = BAR_RED; blk[inner & ~red] = BAR_WHITE
        blk[kp & ((d <= FS) | (d > w5))] = CD.OUT
        rgb[r0:r0 + d.shape[0]] = blk
        n_ok += int(ok.sum())
    print('barriers', n_ok, flush=True)


def fade_edge(rgb, play):
    """За межею зони — м'який перехід у темне: ≈180 м (смугами; відстань рахується на грубій сітці 4 м)."""
    st = max(1, int(round(4 / CD.MPP)))
    pc = np.asarray(play[::st, ::st])
    dc = cv2.distanceTransform((~pc).astype(np.uint8), cv2.DIST_L2, 5) * st * CD.MPP     # метри від межі
    for r0 in range(0, H, 512):
        r1 = min(H, r0 + 512)
        rows = np.clip((np.arange(r0, r1) + 0.5) / st - 0.5, 0, dc.shape[0] - 1).astype(np.float32)
        cols = np.clip((np.arange(W) + 0.5) / st - 0.5, 0, dc.shape[1] - 1).astype(np.float32)
        d = cv2.remap(dc, np.tile(cols, (len(rows), 1)), np.repeat(rows[:, None], W, 1), cv2.INTER_LINEAR)
        t = np.clip(d / 180.0, 0, 1); t = t * t * (3 - 2 * t)
        k = 1 - 0.88 * t
        k[np.asarray(play[r0:r1])] = 1
        blk = np.asarray(rgb[r0:r1]).astype(np.float32) * k[..., None] + np.array(CD.OUT, np.float32) * (1 - k[..., None]) * 0.6
        rgb[r0:r1] = np.clip(blk, 0, 255).astype(np.uint8)


def main(nproc=2):
    print('map', W, H, 'MPP %.4f' % CD.MPP, flush=True)
    st = 16
    mc = zone('margin', 0, H, st)                                  # грубо: які блоки малювати
    jobs = []
    for by in range(0, H, B):
        for bx in range(0, W, B):
            if mc[max(0, by - M) // st:(by + B + M) // st + 1, max(0, bx - M) // st:(bx + B + M) // st + 1].any(): jobs.append((bx, by))
    print('blocks', len(jobs), flush=True)
    # усе велике — на диску (memmap): карта ≈26 500 × 24 000 не вміщається в пам'ять кілька разів
    play = open_memmap('build/play.npy', 'w+', np.bool_, (H, W)); margin = open_memmap('build/margin.npy', 'w+', np.bool_, (H, W))
    for r0 in range(0, H, 1024):
        play[r0:r0 + 1024] = zone('play', r0, min(H, r0 + 1024)); margin[r0:r0 + 1024] = zone('margin', r0, min(H, r0 + 1024))
    if os.environ.get('REUSE') == '1' and os.path.exists('build/rgb_cache.npy') and os.path.exists('build/rgb_cache_ss%d' % SS):
        rgb = np.load('build/rgb_cache.npy', mmap_mode='r+'); cls = np.load('build/cls_cache.npy', mmap_mode='r+')
    else:
        rgb = open_memmap('build/rgb_cache.npy', 'w+', np.uint8, (H, W, 3)); cls = open_memmap('build/cls_cache.npy', 'w+', np.uint8, (H, W))
        for r0 in range(0, H, 1024): rgb[r0:r0 + 1024] = CD.OUT; cls[r0:r0 + 1024] = 0
        render_all(jobs, rgb, cls, nproc)
        rgb.flush(); cls.flush()
        for f_ in os.listdir('build'):
            if f_.startswith('rgb_cache_ss'): os.remove('build/' + f_)
        open('build/rgb_cache_ss%d' % SS, 'w').close()
    out = open_memmap('build/map_rgb.npy', 'w+', np.uint8, (H, W, 3))        # робоча копія (кеш лишається чистим)
    for r0 in range(0, H, 1024):
        blk = np.asarray(rgb[r0:r0 + 1024]).copy(); blk[~np.asarray(margin[r0:r0 + 1024])] = CD.OUT; out[r0:r0 + 1024] = blk
    rgb = out
    barriers(rgb, cls, play)
    # ---- зіткнення: будинки OSM + MS + ручні, вода; поза ігровою зоною — стіна
    D = CD.data()
    solid = open_memmap('build/solid.npy', 'w+', np.uint8, (H, W))
    for b in D['B'] + D.get('SAT', []):
        P = np.array(b['q']) * K
        cv2.fillPoly(solid, [np.round(P).astype(np.int32)], 1)
    for w in D['w']:
        if CD.drivable(w['t']):
            P = np.round(np.array(w['pts']) * K).astype(np.int32)
            cv2.polylines(solid, [P], False, 0, max(2, int(CD.ROAD_W[w['t']['highway']] / CD.MPP)))     # уся ширина: будинки не залазять на дорогу
    print('dropped buildings on roads', D.get('dropped'), flush=True)
    for w in D['w']:                                                   # арки: дорога під будинком — проїзна
        t = w['t']
        if t.get('highway') in CD.ROAD_W and (t.get('tunnel') == 'building_passage' or t.get('covered') == 'yes'):
            P = np.round(np.array(w['pts']) * K).astype(np.int32)
            cv2.polylines(solid, [P], False, 0, max(int(10 * FS), int(CD.ROAD_W[t['highway']] / CD.MPP)))
            for b in D['B'] + D['SAT']:
                Q = np.array(b['q'], np.float32) * K
                if any(cv2.pointPolygonTest(Q, (float(x), float(y)), False) >= 0 for x, y in P.tolist()):
                    ARCH_ROOFS.append(arch_poly(Q))
    seen = set()
    for kind, pts in EXTRA_SOLID:                                      # паркани, сараї, теплиці
        key_ = (kind, tuple(np.round(np.array(pts[:2], float).ravel(), 1)))
        if key_ in seen: continue
        seen.add(key_)
        P = np.round(np.array(pts, float)).astype(np.int32)
        if kind == 'poly': cv2.fillPoly(solid, [P], 1)
        else: cv2.polylines(solid, [P], True, 1, max(2, int(round(2 * FS))))
    print('extra solids', len(seen), flush=True)
    solid.flush()
    sb = open_memmap('build/solid_b.npy', 'w+', np.bool_, (H, W))
    for r0 in range(0, H, 1024): sb[r0:r0 + 1024] = (np.asarray(solid[r0:r0 + 1024]) > 0) | ~np.asarray(play[r0:r0 + 1024])
    sb.flush(); del sb, solid
    os.replace('build/solid_b.npy', 'build/solid.npy')
    solid = np.load('build/solid.npy', mmap_mode='r')
    # ---- палітра карти (для мінікарти й води): лічильник 24-бітних кольорів
    import palgame
    cnt = np.zeros(1 << 24, np.int64)
    key = lambda a: (a[..., 0].astype(np.int32) << 16) | (a[..., 1].astype(np.int32) << 8) | a[..., 2]
    for r0 in range(0, H, 512):
        cnt += np.bincount(key(np.asarray(rgb[r0:r0 + 512])).ravel(), minlength=1 << 24)
    keys = np.nonzero(cnt)[0]
    cols = np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], 1).astype(np.uint8)
    print('unique colors', len(cols), flush=True)
    if len(cols) > palgame.MAP_MAX:
        rs = np.random.RandomState(1)
        samp = cols[rs.choice(len(cols), 400000, p=cnt[keys] / cnt[keys].sum())].astype(np.float32)
        _, _, cen = cv2.kmeans(samp, palgame.MAP_MAX, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2, cv2.KMEANS_PP_CENTERS)
        top = cols[np.argsort(-cnt[keys])[:palgame.MAP_MAX // 2]].astype(np.float32)
        cen = np.vstack([top, cen])[:palgame.MAP_MAX]
        cen = np.unique(np.round(cen).astype(np.int32), axis=0).astype(np.float32)
        lut = np.concatenate([((cols[i:i + 20000, None].astype(np.float32) - cen[None]) ** 2).sum(-1).argmin(1) for i in range(0, len(cols), 20000)])
        pal_cols = np.round(cen).astype(np.uint8)
    else:
        lut = np.arange(len(cols)); pal_cols = cols
    del cnt
    lut24 = np.zeros(1 << 24, np.uint8); lut24[keys] = (lut + palgame.MAP_START).astype(np.uint8)
    json.dump([list(map(int, c)) for c in pal_cols], open('build/map_colors.json', 'w'))
    mp_ = open_memmap('build/map.npy', 'w+', np.uint8, (H, W))
    pc = pal_cols.astype(np.float32)
    dim = np.arange(256, dtype=np.uint8)
    for k, c in enumerate(pc): dim[palgame.MAP_START + k] = palgame.MAP_START + int(((pc - c * 0.72) ** 2).sum(1).argmin())
    for r0 in range(0, H, 512):
        blk = lut24[key(np.asarray(rgb[r0:r0 + 512]))]; o = ~np.asarray(play[r0:r0 + 512]); blk[o] = dim[blk[o]]; mp_[r0:r0 + 512] = blk
    del lut24
    tot = np.zeros(256, np.int64); wat = np.zeros(256, np.int64)                     # які індекси палітри — вода
    for r0 in range(0, H, 1024):
        a = np.asarray(mp_[r0:r0 + 1024]); m_ = (np.asarray(cls[r0:r0 + 1024]) == G_WATER) & np.asarray(play[r0:r0 + 1024])
        tot += np.bincount(a.ravel(), minlength=256); wat += np.bincount(a[m_].ravel(), minlength=256)
    widx = [int(i) for i in range(256) if wat[i] > 500 and wat[i] > 0.8 * tot[i]]
    json.dump(widx, open('build/water_idx.json', 'w'))
    print('water palette ids', len(widx), flush=True)
    mp_.flush(); del mp_
    # ---- повнокольорова карта: за межею — м'який перехід у темне
    fade_edge(rgb, play)
    rgb.flush()
    surf = open_memmap('build/surf.npy', 'w+', np.uint8, (H, W))
    k3 = max(3, int(round(3 * FS)) | 1)
    for r0 in range(0, H, 1024):
        c_ = np.asarray(cls[max(0, r0 - 8):r0 + 1024 + 8]); o0 = r0 - max(0, r0 - 8)
        sf = np.zeros(c_.shape, np.uint8)
        sf[np.isin(c_, (G_GRASS, G_DIRT, G_WOOD, G_SAND, G_PITCH))] = 1
        sf[c_ == G_WATER] = 2
        walk = np.isin(c_, (G_SIDE, G_PED))
        sf[walk] = 3
        sf[walk & (cv2.dilate((c_ == G_ROAD).astype(np.uint8), np.ones((k3, k3), np.uint8)) > 0)] = 4
        surf[r0:r0 + 1024] = sf[o0:o0 + min(1024, H - r0)]
    surf.flush(); del surf
    mapdata(D, cls, solid, play)
    pv = np.asarray(rgb[::5, ::5]).copy()
    cv2.imwrite('build/preview_full.jpg', pv[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])


def mapdata(D, cls, solid, play):
    """Заклади (забір), адреси (доставка), назви вулиць, старт — у сітці гри."""
    import citygen as CG
    MG = CG.MG
    # точки забору/доставки — на проїжджій частині або паркуванні, де вміщається машина
    kk = max(3, int(round(9 * FS * 5.5 / 8.5)) | 1)              # де вміщається машина (5,5 м)
    access = open_memmap('build/access_tmp.npy', 'w+', np.bool_, (H, W))
    for r0 in range(0, H, 1024):
        a0 = max(0, r0 - kk); fr = (~np.asarray(solid[a0:r0 + 1024 + kk]) & np.asarray(play[a0:r0 + 1024 + kk])).astype(np.uint8)
        fr = cv2.erode(fr, np.ones((kk, kk), np.uint8)).astype(bool)[r0 - a0:r0 - a0 + min(1024, H - r0)]
        access[r0:r0 + 1024] = fr & (np.asarray(cls[r0:r0 + 1024]) == G_ROAD)     # лише проїжджа частина

    def gp(p): return (p[0] * K, p[1] * K)

    def stop_at(x, y, maxd=70):
        maxd = int(maxd * FS)
        x, y = int(round(x)), int(round(y))
        if not (40 <= x < W - 40 and 40 <= y < H - 40) or not play[y, x]: return None
        xa, ya = max(0, x - maxd), max(0, y - maxd)
        ys_, xs_ = np.nonzero(access[ya:y + maxd + 1, xa:x + maxd + 1])
        if len(xs_) == 0: return None
        dd = (xs_ + xa - x) ** 2 + (ys_ + ya - y) ** 2; k = dd.argmin()
        if dd[k] > maxd * maxd: return None
        return int(xs_[k] + xa), int(ys_[k] + ya)

    KIND = CG.MG_KIND
    cand = [(t, gp(CG.px(*D['nodes'][nid]))) for nid, t in D['n'].items() if nid in D['nodes']]
    for w in D['w']:
        if w['closed'] and ('shop' in w['t'] or 'amenity' in w['t']):
            P = np.array(w['pts']).mean(0); cand.append((w['t'], gp(P)))
    shops, seen = [], set()
    for t, (x, y) in cand:
        k = t.get('shop') or t.get('amenity')
        if k not in KIND or 'name' not in t: continue
        s = stop_at(x, y)
        if not s: continue
        nm = MG.clean(t['name'])
        if not nm or (nm, s) in seen: continue
        seen.add((nm, s)); shops.append(dict(x=s[0], y=s[1], name=nm, kind=KIND[k]))
    addrs = []
    for b in D['B']:
        t = b['t']
        if 'addr:street' not in t or 'addr:housenumber' not in t: continue
        cx, cy = gp(np.array(b['q']).mean(0))
        s = stop_at(cx, cy, 90)
        if s: addrs.append(dict(x=s[0], y=s[1], name=MG.short_street(t['addr:street']) + ', ' + t['addr:housenumber']))
    names, segs = [], []
    spawn = None
    for w in D['w']:
        h = w['t'].get('highway'); nm = w['t'].get('name')
        if not nm or h not in tuple(CD.ROAD_W) + ('pedestrian',) or w['t'].get('area') == 'yes': continue
        P = [gp(p) for p in w['pts']]
        inside = [play[int(y), int(x)] for x, y in P if 0 <= x < W and 0 <= y < H]
        if not any(inside): continue
        sn = MG.short_street(nm)
        if sn not in names: names.append(sn)
        ni = names.index(sn)
        for i in range(len(P) - 1):
            (xa, ya), (xb, yb) = P[i], P[i + 1]
            if max(xa, xb) < 0 or min(xa, xb) > W or max(ya, yb) < 0 or min(ya, yb) > H: continue
            segs.append((int(xa), int(ya), int(xb), int(yb), ni))
        if spawn is None and 'Проскурівська' in nm:
            spawn = stop_at(*P[len(P) // 2], 200)
    spawn = spawn or (W // 2, H // 2)
    # СТО (евакуація, ремонт)
    sto = []
    cs_ = [(t, gp(CG.px(*D['nodes'][nid]))) for nid, t in D['n'].items() if nid in D['nodes'] and (t.get('shop') == 'car_repair' or t.get('craft') == 'car_repair' or t.get('amenity') == 'car_repair')]
    cs_ += [(w['t'], gp(np.array(w['pts']).mean(0))) for w in D['w'] if w['closed'] and (w['t'].get('shop') == 'car_repair' or w['t'].get('craft') == 'car_repair')]
    for t, (x, y) in cs_:
        s_ = stop_at(x, y, 90)
        if s_ and all(abs(s_[0] - a['x']) + abs(s_[1] - a['y']) > 60 * FS for a in sto):
            sto.append(dict(x=s_[0], y=s_[1], name=MG.clean(t.get('name', 'СТО'))))
    fuel = []
    cf = [(t, gp(CG.px(*D['nodes'][nid]))) for nid, t in D['n'].items() if nid in D['nodes'] and t.get('amenity') == 'fuel']
    cf += [(w['t'], gp(np.array(w['pts']).mean(0))) for w in D['w'] if w['closed'] and w['t'].get('amenity') == 'fuel']
    for t, (x, y) in cf:
        s_ = stop_at(x, y, 90)
        if s_ and all(abs(s_[0] - a['x']) + abs(s_[1] - a['y']) > 60 * FS for a in fuel):
            fuel.append(dict(x=s_[0], y=s_[1], name=MG.clean(t.get('name') or t.get('brand') or 'АЗС')))
    # осьові ліній доріг для помічника смуги: x0,y0,x1,y1, напівширина смуги (пікс), одностороння
    lanes = []
    for w in D['w']:
        t = w['t']; h = t.get('highway')
        if h not in CD.ROAD_W or h in ('track',) or t.get('area') == 'yes': continue
        P = [gp(p) for p in w['pts']]
        if not any(0 <= x < W and 0 <= y < H and play[int(y), int(x)] for x, y in P): continue
        ow = 1 if t.get('oneway') in ('yes', '1') or t.get('junction') == 'roundabout' else (-1 if t.get('oneway') == '-1' else 0)
        nl = 2 if ow == 0 else 1
        try: nl = max(nl, int(t.get('lanes', nl)))
        except ValueError: pass
        lw = CD.ROAD_W[h] / CD.MPP / nl
        for i in range(len(P) - 1):
            a, b = (P[i], P[i + 1]) if ow >= 0 else (P[i + 1], P[i])
            lanes.append((int(a[0]), int(a[1]), int(b[0]), int(b[1]), int(round(lw / 2)), 1 if ow else 0))
    print('sto', len(sto), 'fuel', len(fuel), 'lane segs', len(lanes), 'arch roofs', len(ARCH_ROOFS), flush=True)
    json.dump(dict(shops=shops, addrs=addrs, names=names, segs=segs, spawn=list(spawn), W=W, H=H, M=CD.MPP, sto=sto, lanes=lanes, fuel=fuel, arches=ARCH_ROOFS),
              open('build/mapdata.json', 'w'), ensure_ascii=False)
    print('shops', len(shops), 'addrs', len(addrs), 'streets', len(names), 'segs', len(segs), 'spawn', spawn, flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
