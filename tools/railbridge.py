"""Мости над залізницею (07.11.2026, варіант Б): настил без колій, парапети зі стовпчиками, тінь на коліях, насипи
на підходах. Мости — дороги з bridge=yes над коліями (Західно-Окружна, Кам'янецька, Грушевського, Крамара — колишня Трудова).
decks(D) -> [dict(c=(N,2) вісь, s0, s1, w (пікс напівширина), poly (4,2) прямокутник настилу)]"""
import math
import numpy as np, cv2
import citydraw as CD
from shapely.geometry import LineString

FORCE = ()                       # 07.11.2026: Трудова — це Крамара (у OSM уже міст); на Святослава Хороброго — недіюча колія по дорозі
M = lambda m: m / CD.MPP


def decks(D):
    K = CD.K
    rails = [LineString(np.array(w['pts']) * K) for w in D['w'] if w['t'].get('railway') in ('rail', 'siding', 'narrow_gauge')
             and w['t'].get('tunnel') != 'yes' and len(w['pts']) > 1]
    from shapely.strtree import STRtree
    tree = STRtree(rails)
    out = []
    for w in D['w']:
        t = w['t']; h = t.get('highway')
        if h not in CD.ROAD_W or len(w['pts']) < 2: continue
        if not (t.get('bridge') in ('yes', 'viaduct') or any(f in t.get('name', '') for f in FORCE)): continue
        L = LineString(np.array(w['pts']) * K)
        ss = []
        for i in tree.query(L):
            r = rails[int(i)]
            if not L.intersects(r): continue
            p = L.intersection(r)
            for g in (getattr(p, 'geoms', None) or [p]):
                c = g.centroid; ss.append(L.project(c))
        if not ss: continue
        ss.sort()
        grp = [[ss[0]]]
        for v in ss[1:]:
            if v - grp[-1][-1] < M(60): grp[-1].append(v)
            else: grp.append([v])
        hw = CD.ROAD_W[h] / 2 / CD.MPP
        near_r = [rails[int(i)] for i in tree.query(L.buffer(hw + M(3)))]
        def covered(v):                                                   # колія під дорогою в цьому місці (з запасом на ширину)
            pt = L.interpolate(v)
            return any(r.distance(pt) < hw + M(1.5) for r in near_r)
        for g in grp:
            s0, s1 = g[0], g[-1]
            while s0 > 0 and covered(s0 - M(1)): s0 -= M(1)                   # гострий кут: колія довго йде під дорогою
            while s1 < L.length and covered(s1 + M(1)): s1 += M(1)
            s0, s1 = max(0, s0 - M(4)), min(L.length, s1 + M(4))
            pa, pb = L.interpolate(s0), L.interpolate(s1)
            ux, uy = pb.x - pa.x, pb.y - pa.y; n = math.hypot(ux, uy) or 1; ux /= n; uy /= n
            nx, ny = -uy, ux
            poly = np.array([(pa.x + nx * hw, pa.y + ny * hw), (pb.x + nx * hw, pb.y + ny * hw), (pb.x - nx * hw, pb.y - ny * hw), (pa.x - nx * hw, pa.y - ny * hw)])
            out.append(dict(line=L, s0=s0, s1=s1, hw=hw, poly=poly, u=(ux, uy), name=t.get('name', ''), rails=near_r))
    print('  мости над коліями: %d (%s)' % (len(out), ', '.join(sorted({d['name'] for d in out}))))
    return out


def paint(rgb, solid, surf, dk, org=(0, 0)):
    """малює один міст на карті (rgb, solid — memmap, змінюються на місці; surf — 5 під настилом).
    Локальна система координат іде вздовж осі дороги (вигнуті мости теж): al — уздовж від початку настилу, ac — убік."""
    H, W = rgb.shape[:2]
    L, hw, s0, s1 = dk['line'], dk['hw'], dk['s0'], dk['s1']
    ext = M(26); dl = s1 - s0
    a0, a1 = max(0.0, s0 - ext - M(4)), min(L.length, s1 + ext + M(4))
    ss = np.arange(a0, a1 + 1e-6, M(2.0))
    Q = np.array([L.interpolate(v).coords[0] for v in ss]) - np.array(org, np.float64)
    Aq = ss - s0                                                            # дуга вузлів відносно початку настилу
    x0, y0 = np.floor(Q.min(0) - hw - M(14)).astype(int); x1, y1 = np.ceil(Q.max(0) + hw + M(14)).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    img0 = np.asarray(rgb[y0:y1, x0:x1]).astype(np.float32); img = img0.copy(); sl = np.asarray(solid[y0:y1, x0:x1]).copy()
    hh, ww = img.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32); xx += x0; yy += y0
    best = np.full((hh, ww), 1e18, np.float32); al = np.zeros((hh, ww), np.float32); ac = np.zeros((hh, ww), np.float32)
    for j in range(len(Q) - 1):
        ax, ay = Q[j]; dx, dy = Q[j + 1] - Q[j]; l2 = dx * dx + dy * dy
        if l2 < 1e-6: continue
        t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / l2, 0, 1)
        px, py = ax + dx * t, ay + dy * t; d2 = (xx - px) ** 2 + (yy - py) ** 2
        m = d2 < best; best[m] = d2[m]
        ln = math.sqrt(l2)
        al[m] = (Aq[j] + t * ln)[m]; ac[m] = (((xx - px) * (-dy) + (yy - py) * dx) / ln)[m]
    # положення й нормаль осі за дугою (для вибірки текстури)
    nrm = np.zeros_like(Q); d_ = np.gradient(Q, axis=0); d_ /= np.maximum(1e-6, np.hypot(d_[:, 0], d_[:, 1]))[:, None]
    nrm[:, 0], nrm[:, 1] = -d_[:, 1], d_[:, 0]
    def at(a, c):
        qx = np.interp(a, Aq, Q[:, 0]); qy = np.interp(a, Aq, Q[:, 1]); nx_ = np.interp(a, Aq, nrm[:, 0]); ny_ = np.interp(a, Aq, nrm[:, 1])
        return (qx + nx_ * c - x0).astype(np.float32), (qy + ny_ * c - y0).astype(np.float32)
    deck = (al >= 0) & (al <= dl) & (np.abs(ac) <= hw) & (best < (hw + M(1)) ** 2)
    # 1) настил (дорога й тротуари мосту): колії стерти — домалювати з сусідніх пікселів (розмітка лишається)
    med = np.array([70, 70, 74], np.float32)
    if deck.any(): med = np.median(img0[deck], 0)
    # 2) тінь настилу на коліях (сонце з північного заходу): та сама смуга, зсунута на (1,4; 2,2) м
    sx, sy = at(np.clip(al, 0, dl), np.clip(ac, -hw, hw))
    shx, shy = xx - M(1.4), yy - M(2.2)
    # піксель у тіні, якщо точка (x - зсув) лежить на настилі
    mx_, my_ = (shx - x0).astype(np.float32), (shy - y0).astype(np.float32)
    sh_al = cv2.remap(al, mx_, my_, cv2.INTER_NEAREST, borderValue=-1e9)
    sh_ac = cv2.remap(ac, mx_, my_, cv2.INTER_NEAREST, borderValue=1e9)
    shadow = (sh_al >= 0) & (sh_al <= dl) & (np.abs(sh_ac) <= hw + M(0.7)) & ~deck & ~((np.abs(ac) <= hw + M(1.0)) & (best < (hw + M(2)) ** 2))   # не на саму дорогу
    if deck.any():                                                          # тінь не лягає на асфальт (дорога там на насипі)
        shadow &= np.abs(img0 - med).sum(-1) > 36
    img[shadow] *= 0.58
    # 3) насипи на підходах: клин уздовж дороги за бортом, від настилу назовні
    side = np.abs(ac) - hw
    near = best < (hw + M(6)) ** 2
    for lo, hi in ((-ext, 0), (dl, dl + ext)):
        z = (al > lo) & (al < hi) & near
        t = np.where(al < 0, -al, al - dl) / ext
        wedge = z & (side > M(0.75)) & (side < M(0.75) + M(4.5) * (1 - t))
        k = (0.5 * (1 - t))[wedge][:, None]
        img[wedge] = img[wedge] * (1 - k) + np.array([132, 128, 104], np.float32) * k
    # 4) парапети з обох боків (настил + підходи), стовпчики огорожі
    span = (al > -ext * 0.7) & (al < dl + ext * 0.7) & near
    par = span & (np.abs(side - M(0.35)) < M(0.3))
    edge = span & (np.abs(side - M(0.35)) < M(0.45)) & ~par
    img[edge] = img[edge] * 0.45 + np.array([60, 60, 62]) * 0.55
    img[par] = np.array([206, 206, 200], np.float32)
    post = par & (np.mod(al, M(2.5)) < M(0.35))
    img[post] = np.array([96, 96, 100], np.float32)
    sl[span & (np.abs(side - M(0.35)) < M(0.45))] = 1                     # з мосту не з'їхати
    sl[deck] = 0
    rgb[y0:y1, x0:x1] = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    solid[y0:y1, x0:x1] = sl
    if surf is not None:
        sf = np.asarray(surf[y0:y1, x0:x1]).copy(); sf[deck] = 5; surf[y0:y1, x0:x1] = sf
    dk['mask_box'] = (x0 + org[0], y0 + org[1])
    return int(deck.sum())
