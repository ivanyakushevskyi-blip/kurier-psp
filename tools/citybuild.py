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
B, M = 1024, 64
LAT0, LON0 = 49.422, 26.995
MLAT = 111320.0; MLON = 111320.0 * math.cos(math.radians(LAT0))
G_YARD, G_GRASS, G_WATER, G_SIDE, G_ROAD, G_PATH, G_PED, G_RAIL, G_DIRT, G_PITCH, G_TRACK, G_PARK, G_PAVED, G_WOOD, G_SAND = range(1, 16)


def zone(key):
    """play / margin з build/bounds.npz у сітці гри."""
    b = np.load('build/bounds.npz')
    z = b[key]; G = float(b['G'])
    lon0, lat1, mlon, mlat = float(b['lon0']), float(b['lat1']), float(b['mlon']), float(b['mlat'])
    xs = META['x0'] + (np.arange(W) + 0.5) * CD.MPP; ys = META['y0'] + (np.arange(H) + 0.5) * CD.MPP
    lon = LON0 + xs / MLON; lat = LAT0 - ys / MLAT
    jx = np.clip(((lon - lon0) * mlon / G).astype(int), 0, z.shape[1] - 1)
    iy = np.clip(((lat1 - lat) * mlat / G).astype(int), 0, z.shape[0] - 1)
    return z[iy[:, None], jx[None, :]]


def render_block(args):
    bx, by = args
    w, h = min(B, W - bx), min(B, H - by)
    tl = []
    img, c = CD.render(bx - M, by - M, w + 2 * M, h + 2 * M, tree_out=tl)
    trees = [t for t in tl if bx <= t[0] < bx + w and by <= t[1] < by + h]
    return bx, by, img[M:M + h, M:M + w].copy(), c[M:M + h, M:M + w].copy(), trees


def render_all(jobs, rgb, cls, nproc):
    trees = []
    CD.data()                                    # кеш OSM/будинків до розгалуження процесів
    if nproc > 1:
        import multiprocessing as mp
        pool = mp.get_context('fork').Pool(nproc, maxtasksperchild=8)
        it = pool.imap_unordered(render_block, jobs)
    else:
        it = map(render_block, jobs)
    for k, (bx, by, img, c, tl) in enumerate(it):
        h, w = c.shape
        rgb[by:by + h, bx:bx + w] = img; cls[by:by + h, bx:bx + w] = c; trees.extend(tl)
        print('  block %d/%d' % (k + 1, len(jobs)), bx, by, flush=True)
    if nproc > 1: pool.close(); pool.join()
    # ---- дерева одним проходом (без швів між блоками)
    for x, y, rr, kind in sorted(trees, key=lambda t: (t[1], t[0])):
        CD.draw_tree(rgb, x, y, rr, kind, random.Random(x * 7919 + y))
    print('trees', len(trees), flush=True)


BAR_RED, BAR_WHITE = (200, 56, 44), (236, 236, 230)


def barriers(rgb, cls, play):
    """Межа зони на дорогах: червоно-білий бетонний блок поперек проїзду (смуга 5 пікс уздовж межі).
    Лише там, де дорога перетинає межу; де дорога йде вздовж межі — нічого (довгі смуги відкидаються)."""
    dist = cv2.distanceTransform(play.astype(np.uint8), cv2.DIST_L2, 3)
    road = cv2.dilate((cls == G_ROAD).astype(np.uint8), np.ones((5, 5), np.uint8))
    band = (road > 0) & (dist > 0) & (dist <= 6)
    n, lab, st, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
    ok = np.zeros(n, bool)
    ok[1:] = (st[1:, cv2.CC_STAT_AREA] >= 20) & (st[1:, cv2.CC_STAT_AREA] <= 420) & \
             (np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) <= 70)
    keep = ok[lab]; del lab, band
    for r0 in range(0, H, 1024):
        sl = slice(r0, r0 + 1024)
        d = dist[sl]; kp = keep[sl]
        yy, xx = np.mgrid[r0:r0 + d.shape[0], 0:W]
        red = ((xx + yy) // 6) % 2 == 0
        blk = rgb[sl]
        inner = kp & (d > 1) & (d <= 5)
        blk[inner & red] = BAR_RED; blk[inner & ~red] = BAR_WHITE
        blk[kp & ((d <= 1) | (d > 5))] = CD.OUT
    print('barriers', int(ok.sum()), flush=True)


def main(nproc=2):
    print('map', W, H, flush=True)
    margin = zone('margin')
    rgb = np.empty((H, W, 3), np.uint8); rgb[:] = CD.OUT
    cls = np.zeros((H, W), np.uint8)
    jobs = []
    for by in range(0, H, B):
        for bx in range(0, W, B):
            if margin[max(0, by - M):by + B + M, max(0, bx - M):bx + B + M].any(): jobs.append((bx, by))
    print('blocks', len(jobs), flush=True)
    if os.environ.get('REUSE') == '1' and os.path.exists('build/rgb_cache.npy'):
        rgb = np.load('build/rgb_cache.npy'); cls = np.load('build/cls_cache.npy')    # лише пост-обробка (межа, палітра, mapdata)
    else:
        render_all(jobs, rgb, cls, nproc)
        np.save('build/rgb_cache.npy', rgb); np.save('build/cls_cache.npy', cls)
    # поза запасом — темне тло
    rgb[~margin] = CD.OUT
    play = zone('play')
    barriers(rgb, cls, play)
    # ---- зіткнення: будинки OSM + MS + ручні, вода; поза ігровою зоною — стіна
    D = CD.data()
    solid = np.zeros((H, W), np.uint8)
    for b in D['B'] + D.get('SAT', []):
        P = np.array(b['q']) * K
        cv2.fillPoly(solid, [np.round(P).astype(np.int32)], 1)
    solid = solid.astype(bool) | (cls == G_WATER)
    solid |= ~play
    # ---- палітра карти: лічильник 24-бітних кольорів
    import palgame
    cnt = np.zeros(1 << 24, np.int64)
    key = lambda a: (a[..., 0].astype(np.int32) << 16) | (a[..., 1].astype(np.int32) << 8) | a[..., 2]
    for r0 in range(0, H, 512):
        cnt += np.bincount(key(rgb[r0:r0 + 512]).ravel(), minlength=1 << 24)
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
    mp_ = np.empty((H, W), np.uint8)
    for r0 in range(0, H, 512): mp_[r0:r0 + 512] = lut24[key(rgb[r0:r0 + 512])]
    del lut24
    # поза ігровою зоною — неяскраво приглушено (той самий колір ×0,72, найближчий у палітрі)
    pc = pal_cols.astype(np.float32)
    dim = np.arange(256, dtype=np.uint8)
    for k, c in enumerate(pc): dim[palgame.MAP_START + k] = palgame.MAP_START + int(((pc - c * 0.72) ** 2).sum(1).argmin())
    for r0 in range(0, H, 512):
        o = ~play[r0:r0 + 512]; blk = mp_[r0:r0 + 512]; blk[o] = dim[blk[o]]
    np.save('build/map.npy', mp_); np.save('build/solid.npy', solid); np.save('build/play.npy', play)
    del mp_
    mapdata(D, cls, solid, play)
    pv = rgb[::3, ::3].copy(); pl = play[::3, ::3]; pv[~pl] = (pv[~pl] * 0.5).astype(np.uint8)
    cv2.imwrite('build/preview_full.jpg', pv[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])


def mapdata(D, cls, solid, play):
    """Заклади (забір), адреси (доставка), назви вулиць, старт — у сітці гри."""
    import citygen as CG
    MG = CG.MG
    access = np.isin(cls, (G_SIDE, G_PATH, G_PED, G_PARK, G_PAVED)) & ~solid & play

    def gp(p): return (p[0] * K, p[1] * K)

    def stop_at(x, y, maxd=70):
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
    json.dump(dict(shops=shops, addrs=addrs, names=names, segs=segs, spawn=list(spawn), W=W, H=H, M=CD.MPP),
              open('build/mapdata.json', 'w'), ensure_ascii=False)
    print('shops', len(shops), 'addrs', len(addrs), 'streets', len(names), 'segs', len(segs), 'spawn', spawn, flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
